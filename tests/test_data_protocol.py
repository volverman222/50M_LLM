import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('llm_mini_lab.data_protocol'), 'Configurable data protocol missing'
    from llm_mini_lab import data_protocol
    return data_protocol


class Encoder:
    eot_token = 99
    def encode(self, text):
        return [ord(c) % 80 for c in text]


class Stream:
    def __init__(self, rows): self.rows = list(rows)
    def __iter__(self): return iter(self.rows)
    def filter(self, fn): return Stream([r for r in self.rows if fn(r)])
    def shuffle(self, seed, buffer_size):
        import random
        rows = list(self.rows); random.Random(seed).shuffle(rows)
        return Stream(rows)


def config(tmp_path):
    return dict(schema_version=1, protocol_id='fixture',
        source=dict(backend='local_jsonl', files={'train':str(tmp_path/'train.jsonl')},
                    hashes={'train':'a'*64}, text_field='text'),
        tokenizer=dict(name='sp16384', sha256='b'*64),
        sequence=dict(length=4, packing='packed', add_eos=True, drop_tail=True),
        split=dict(strategy='stable_hash', modulus=10, validation_buckets=[0], test_buckets=[1], seed=5),
        order=dict(seed=123, shuffle_buffer=10, reseed_each_epoch=True),
        batch=dict(size=2, drop_last=True), num_workers=0)


def test_unknown_fields_nonpositive_length_and_overlapping_buckets_rejected(tmp_path):
    m=api()
    for mutate in [lambda p:p.update(unknown=True),lambda p:p['sequence'].update(length=0),
                   lambda p:p['split'].update(test_buckets=[0]),lambda p:p.update(num_workers=2),
                   lambda p:p['split'].update(modulus=1),lambda p:p['split'].update(validation_buckets=[]),
                   lambda p:p['sequence'].update(packing='unknown')]:
        p=config(tmp_path);mutate(p)
        with pytest.raises(ValueError):m.validate(p)


def test_source_hf_requires_immutable_revision(tmp_path):
    m=api();p=config(tmp_path)
    p['source']=dict(backend='huggingface',dataset='HuggingFaceTB/smollm-corpus',config='cosmopedia-v2',
                     revision='main',split='train',text_field='text')
    with pytest.raises(ValueError,match='revision'):m.validate(p)
    p['source']['revision']='c'*40
    assert m.validate(p)['source']['revision']=='c'*40


def test_hash_membership_independent_of_order_seed_order_and_epoch(tmp_path):
    m=api();p=config(tmp_path);m.validate(p)
    rows=[{'text':f'document {i:04d}'} for i in range(1000)]
    original={r['text']:m.partition(r,p) for r in rows}
    q=copy.deepcopy(p);q['order']['seed']=98765
    assert original=={r['text']:m.partition(r,q) for r in reversed(rows)}
    assert set(original.values())=={'train','validation','test'}
    for role in ('train','validation','test'):
        first=list(m.select_rows(Stream(rows),p,role,epoch=0))
        second=list(m.select_rows(Stream(rows),q,role,epoch=3))
        assert {r['text'] for r in first}=={r['text'] for r in second}


def test_legacy_modulo_matches_upstream_document_selection(tmp_path):
    m=api();p=config(tmp_path);p['split']['strategy']='upstream_modulo'
    p['order']['reseed_each_epoch']=False
    rows=[{'text':f'doc {i}'} for i in range(100)]+[{'text':''}]
    shuffled=list(Stream(rows).shuffle(123,10))
    valid=[r for r in shuffled if r['text'].strip()]
    for role in ('train','validation','test'):
        expected=[r for i,r in enumerate(valid) if m.bucket_role(i%10,p)==role]
        assert list(m.select_rows(Stream(rows),p,role))==expected
    p['order']['reseed_each_epoch']=True
    with pytest.raises(ValueError):m.validate(p)


def test_named_splits_are_distinct_and_not_resplit(tmp_path):
    m=api();p=config(tmp_path)
    p['split']=dict(strategy='source_splits',names={'train':'train','validation':'validation','test':'test'})
    p['source']['files']={s:str(tmp_path/(s+'.jsonl')) for s in ('train','validation','test')}
    p['source']['hashes']={s:'a'*64 for s in p['source']['files']}
    m.validate(p)
    rows=[{'text':'first'},{'text':'second'}]
    assert list(m.select_rows(Stream(rows),p,'validation'))==rows
    p['split']['names']['test']='validation'
    with pytest.raises(ValueError):m.validate(p)


def test_packing_eos_shift_and_document_isolation(tmp_path):
    m=api();p=config(tmp_path);enc=Encoder();rows=[{'text':'abcdef'},{'text':'ghijkl'}]
    pairs=list(m.token_pairs(rows,enc,p))
    full=enc.encode('abcdef')+[99]+enc.encode('ghijkl')+[99]
    assert pairs==[(full[i:i+4],full[i+1:i+5]) for i in range(0,len(full)-4,4)]
    p['sequence']['packing']='document'
    pairs=list(m.token_pairs(rows,enc,p))
    assert pairs==[(enc.encode('abcdef')[:4],enc.encode('abcdef')[1:5]),
                   (enc.encode('ghijkl')[:4],enc.encode('ghijkl')[1:5])]


def test_local_integrity_and_exact_hf_call(tmp_path):
    m=api();p=config(tmp_path);path=Path(p['source']['files']['train'])
    path.write_text('{"text":"document"}\n',encoding='utf-8');p['source']['hashes']['train']=m.sha256_file(path)
    assert list(m.source_stream(p,'train'))==[{'text':'document'}]
    path.write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError,match='hash'):m.source_stream(p,'train')
    p['source']=dict(backend='huggingface',dataset='codelion/fineweb-edu-1B',config=None,
                    revision='c'*40,split='train',text_field='text')
    calls=[]
    def loader(*args,**kw):calls.append((args,kw));return Stream([{'text':'abc'}])
    list(m.source_stream(p,'train',loader=loader))
    assert calls==[(('codelion/fineweb-edu-1B',),dict(name=None,revision='c'*40,split='train',streaming=True))]


def test_receipt_has_protocol_hash_units_order_and_limitations(tmp_path):
    m=api();p=config(tmp_path);receipt=m.receipt(p)
    assert receipt['protocol_sha256']==m.fingerprint(p)
    assert receipt['sequence_length']==4
    assert receipt['split_before_order'] is True
    assert receipt['decontamination_verified'] is False
    assert receipt['test_used_for_training'] is False


def test_cli_default_unchanged_and_opt_in_flag(tmp_path, monkeypatch):
    api();sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import train_pretrain_1b as trainer
    monkeypatch.setattr(sys,'argv',['train','--dry-run'])
    assert trainer.parse_args().context_length==256
    monkeypatch.setattr(sys,'argv',['train','--data-protocol',str(tmp_path/'p.json'),'--dry-run'])
    assert trainer.parse_args().data_protocol==tmp_path/'p.json'


def test_missing_text_field_fails_not_empty_training(tmp_path):
    m=api();p=config(tmp_path)
    with pytest.raises(ValueError,match='text'):
        list(m.select_rows(Stream([{'other':'data'}]),p,'train'))


def test_loader_validates_cli_shape_and_seed_is_not_split_seed(tmp_path):
    m=api();p=config(tmp_path)
    from types import SimpleNamespace
    args=SimpleNamespace(context_length=8,micro_batch_size=2,num_workers=0,seed=111,tokenizer='sp16384',tokenizer_model=None)
    with pytest.raises(ValueError,match='context'):m.validate_runtime(p,args)


def test_training_loader_factory_never_opens_the_test_role(tmp_path,monkeypatch):
    m=api();p=config(tmp_path)
    p['tokenizer']={'name':'gpt2','sha256':None}
    from types import SimpleNamespace
    args=SimpleNamespace(context_length=4,micro_batch_size=2,num_workers=0,
        tokenizer='gpt2',tokenizer_model=None,resume=None,data_protocol=tmp_path/'p.json',_data_contract=p)
    calls=[]
    monkeypatch.setattr(m,'torch_loader',lambda protocol,role,encoder,epoch=0: calls.append((role,epoch)) or role)
    assert m.protocol_loaders(args,Encoder(),epoch=3)==('train','validation')
    assert calls==[('train',3),('validation',0)]


def test_fixed_validation_identity_hashes_only_scored_pairs_without_rng_draws():
    m=api()
    import torch,hashlib
    from torch.utils.data import DataLoader
    from llm_mini_lab.training.core import FixedPairsDataset
    pairs=[(torch.tensor([i,i+1]),torch.tensor([i+1,i+2])) for i in range(5)]
    loader=DataLoader(FixedPairsDataset(pairs),batch_size=2,drop_last=True)
    before=torch.get_rng_state().clone()
    identity=m.fixed_validation_identity(loader,max_batches=20)
    expected=hashlib.sha256()
    for x,y in pairs[:4]:
        expected.update(x.numpy().astype('<i8',copy=False).tobytes())
        expected.update(y.numpy().astype('<i8',copy=False).tobytes())
    assert identity['sha256']==expected.hexdigest()
    assert identity['examples']==4 and identity['target_tokens']==8
    assert identity['batches']==2 and identity['sequence_lengths']==[2]
    assert torch.equal(before,torch.get_rng_state())


def test_empty_fixed_validation_cannot_be_reported_as_a_valid_check():
    m=api()
    from torch.utils.data import DataLoader
    from llm_mini_lab.training.core import FixedPairsDataset
    with pytest.raises(ValueError,match='validation'):
        m.fixed_validation_identity(DataLoader(FixedPairsDataset([]),batch_size=2),max_batches=20)
