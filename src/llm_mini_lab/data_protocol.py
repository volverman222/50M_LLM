"""Explicit, opt-in data protocol. Legacy loaders/defaults remain unchanged.

Partition identity is separate from traversal order. No claim of semantic
near-duplicate decontamination is made by content hashes or named splits.
"""
from __future__ import annotations
import copy
import hashlib
import json
import random
import re
from collections import deque
from pathlib import Path

ROLES = ('train', 'validation', 'test')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(required)-set(value) or set(value)-set(required)-set(optional):
        raise ValueError(f'Missing/unknown fields: required={required}; got={list(value) if isinstance(value,dict) else type(value)}')


def _integer(x, minimum=0):
    return type(x) is int and x >= minimum


def validate(config):
    p=copy.deepcopy(config)
    _keys(p, ['schema_version','protocol_id','source','tokenizer','sequence','split','order','batch','num_workers'], ['limits'])
    if type(p['schema_version']) is not int or p['schema_version'] != 1 or not isinstance(p['protocol_id'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',p['protocol_id']):
        raise ValueError('Invalid protocol identity/version')
    s=p['source']
    if not isinstance(s,dict): raise ValueError('Source must be an object')
    if s.get('backend')=='huggingface':
        _keys(s,['backend','dataset','config','revision','split','text_field'])
        if not isinstance(s['dataset'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',s['dataset']):
            raise ValueError('Use an explicit owner/dataset repository ID')
        if not isinstance(s['revision'],str) or not re.fullmatch('[0-9a-f]{40}',s['revision']):
            raise ValueError('Dataset revision must be an immutable 40-character commit SHA')
        if s['config'] is not None and (not isinstance(s['config'],str) or not s['config']):
            raise ValueError('Dataset configuration must be null or a nonempty string')
        if not isinstance(s['split'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',s['split']):
            raise ValueError('Only exact native split names, not slice expressions, are supported')
    elif s.get('backend')=='local_jsonl':
        _keys(s,['backend','files','hashes','text_field'])
        if not isinstance(s['files'],dict) or not s['files'] or not isinstance(s['hashes'],dict) or set(s['hashes'])!=set(s['files']):
            raise ValueError('Every local split file requires its own SHA256')
        for key,path in s['files'].items():
            if not isinstance(path,str) or not path or not isinstance(s['hashes'][key],str) or not re.fullmatch('[0-9a-f]{64}',s['hashes'][key]):
                raise ValueError('Invalid local path/hash')
    else: raise ValueError('Unsupported source backend')
    if not isinstance(s['text_field'],str) or not s['text_field']: raise ValueError('text_field is required')
    t=p['tokenizer'];_keys(t,['name','sha256'])
    if t['name'] not in ('gpt2','sp16384'): raise ValueError('Unsupported tokenizer')
    if t['name']=='sp16384' and (not isinstance(t['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',t['sha256'])):
        raise ValueError('SP16K requires the tokenizer model SHA256')
    if t['name']=='gpt2' and t['sha256'] is not None: raise ValueError('GPT2 identity uses tiktoken version; sha256 must be null')
    q=p['sequence'];_keys(q,['length','packing','add_eos','drop_tail'])
    if not _integer(q['length'],2) or q['length']>65536: raise ValueError('Invalid context sequence length')
    if q['packing'] not in ('packed','document'): raise ValueError('Unsupported sequence packing')
    if type(q['add_eos']) is not bool or q['drop_tail'] is not True: raise ValueError('EOS must be boolean; only explicit tail drop supported')
    b=p['batch'];_keys(b,['size','drop_last'])
    if not _integer(b['size'],1) or type(b['drop_last']) is not bool: raise ValueError('Invalid batch')
    if type(p['num_workers']) is not int or p['num_workers']!=0:
        raise ValueError('Only num_workers=0 supported until shard/split equivalence is verified')
    o=p['order'];_keys(o,['seed','shuffle_buffer','reseed_each_epoch'])
    if not _integer(o['seed']) or not _integer(o['shuffle_buffer']) or type(o['reseed_each_epoch']) is not bool:
        raise ValueError('Invalid data order settings')
    sp=p['split']
    if not isinstance(sp,dict): raise ValueError('split must be an object')
    if sp.get('strategy')=='source_splits':
        _keys(sp,['strategy','names']);names=sp['names'];_keys(names,['train','validation','test'])
        values=[v for v in names.values() if v is not None]
        if names['train'] is None or names['validation'] is None or len(set(values))!=len(values):
            raise ValueError('Native split names must be distinct; training and validation required')
        if any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',v) for v in values): raise ValueError('Invalid native split names')
        if s['backend']=='local_jsonl' and (set(values)-set(s['files']) or len({str(Path(s['files'][v]).resolve()) for v in values})!=len(values)):
            raise ValueError('Named roles need distinct local files')
    elif sp.get('strategy') in ('stable_hash','upstream_modulo'):
        _keys(sp,['strategy','modulus','validation_buckets','test_buckets','seed'])
        if not _integer(sp['modulus'],2) or not _integer(sp['seed']): raise ValueError('Invalid split modulus/seed')
        for key in ['validation_buckets','test_buckets']:
            if not isinstance(sp[key],list) or any(not _integer(x) or x>=sp['modulus'] for x in sp[key]) or len(set(sp[key]))!=len(sp[key]):
                raise ValueError('Invalid split buckets')
        if not sp['validation_buckets'] or set(sp['validation_buckets'])&set(sp['test_buckets']) or len(sp['validation_buckets'])+len(sp['test_buckets'])>=sp['modulus']:
            raise ValueError('Split buckets overlap, lack validation, or consume training')
        if s['backend']=='local_jsonl' and set(s['files'])!={'train'}: raise ValueError('Partitioned local source requires a single train file')
        if sp['strategy']=='upstream_modulo' and o['reseed_each_epoch']:
            raise ValueError('Legacy modulo cannot reshuffle each epoch without changing membership')
    else: raise ValueError('Unsupported split strategy')
    training_name=sp['names']['train'] if sp['strategy']=='source_splits' else s.get('split','train')
    if training_name.casefold() in {'test','validation','valid','val','dev'}:
        raise ValueError('A held-out split cannot be configured as the training source')
    limits=p.get('limits',{})
    if not isinstance(limits,dict) or set(limits)-set(ROLES) or any(v is not None and not _integer(v,1) for v in limits.values()):
        raise ValueError('Limits are positive document counts by role')
    return p


def fingerprint(p):
    return hashlib.sha256(json.dumps(validate(p),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def bucket_role(bucket,p):
    sp=p['split']
    return 'validation' if bucket in sp['validation_buckets'] else 'test' if bucket in sp['test_buckets'] else 'train'


def partition(row,p):
    text=row[p['source']['text_field']]
    payload=str(p['split']['seed']).encode()+b'\0'+text.encode('utf-8')
    bucket=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%p['split']['modulus']
    return bucket_role(bucket,p)


class LocalRows:
    """Restartable local JSONL with bounded buffered shuffle (no global RNG)."""
    def __init__(self,path,transforms=()): self.path,self.transforms=Path(path),transforms
    def _rows(self):
        with self.path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    def __iter__(self):
        rows=self._rows()
        for kind,args in self.transforms:
            if kind=='filter': rows=filter(args,rows)
            else: rows=_shuffle(rows,*args)
        return iter(rows)
    def filter(self,fn): return LocalRows(self.path,self.transforms+(('filter',fn),))
    def shuffle(self,seed,buffer_size): return LocalRows(self.path,self.transforms+(('shuffle',(seed,buffer_size)),))


def _shuffle(rows,seed,size):
    rng=random.Random(seed);buffer=[]
    for row in rows:
        if len(buffer)<size: buffer.append(row);continue
        i=rng.randrange(size);yield buffer[i];buffer[i]=row
    rng.shuffle(buffer);yield from buffer


def source_stream(p,role,loader=None):
    p=validate(p)
    if role not in ROLES: raise ValueError('Unknown split role')
    s,sp=p['source'],p['split']
    native=sp['names'][role] if sp['strategy']=='source_splits' else s.get('split','train')
    if native is None: raise ValueError(f'{role} not configured')
    if s['backend']=='local_jsonl':
        path=Path(s['files'][native])
        if sha256_file(path)!=s['hashes'][native]: raise ValueError('Local source hash mismatch')
        return LocalRows(path)
    if loader is None:
        from datasets import load_dataset
        loader=load_dataset
    return loader(s['dataset'],name=s['config'],revision=s['revision'],split=native,streaming=True)


def select_rows(stream,p,role,epoch=0):
    if role not in ROLES: raise ValueError('Unknown split role')
    sp,o=p['split'],p['order'];field=p['source']['text_field']
    def nonempty(row):
        if not isinstance(row,dict) or field not in row: raise ValueError(f'Missing text column: {field}')
        text=row[field]
        return isinstance(text,str) and bool(text.strip())
    seed=o['seed']+(epoch if role=='train' and o['reseed_each_epoch'] else 0)
    if sp['strategy']=='upstream_modulo':
        if o['shuffle_buffer']:stream=stream.shuffle(seed=o['seed'],buffer_size=o['shuffle_buffer'])
        valid=filter(nonempty,stream)
        selected=(row for index,row in enumerate(valid) if bucket_role(index%sp['modulus'],p)==role)
    else:
        stream=stream.filter(nonempty)
        if sp['strategy']=='stable_hash':stream=stream.filter(lambda row:partition(row,p)==role)
        if role=='train' and o['shuffle_buffer']:stream=stream.shuffle(seed=seed,buffer_size=o['shuffle_buffer'])
        selected=iter(stream)
    limit=p.get('limits',{}).get(role)
    for index,row in enumerate(selected):
        if limit is not None and index>=limit: break
        yield row


def token_pairs(rows,encoder,p):
    seq=p['sequence'];length=seq['length'];buf=deque()
    if seq['add_eos']:
        eos=encoder.eot_token if hasattr(encoder,'eot_token') else encoder.eos_id()
        if eos is None or eos<0: raise ValueError('Tokenizer has no EOS token')
    for row in rows:
        tokens=list(encoder.encode(row[p['source']['text_field']]))
        if seq['add_eos']:tokens.append(eos)
        if seq['packing']=='document':buf.clear()
        buf.extend(tokens)
        while len(buf)>=length+1:
            x=[buf.popleft() for _ in range(length)]
            yield x,x[1:]+[buf[0]]


def validate_runtime(p,args):
    validate(p)
    if args.context_length!=p['sequence']['length']: raise ValueError('CLI context_length differs from protocol')
    if args.micro_batch_size!=p['batch']['size'] or args.num_workers!=p['num_workers']:
        raise ValueError('CLI batch/workers differ from protocol')
    if args.tokenizer!=p['tokenizer']['name']: raise ValueError('CLI tokenizer differs from protocol')
    if args.tokenizer=='sp16384' and (args.tokenizer_model is None or sha256_file(args.tokenizer_model)!=p['tokenizer']['sha256']):
        raise ValueError('Tokenizer model hash mismatch')
    if getattr(args,'resume',None) is not None:
        raise ValueError('Protocol resume requires a verified data cursor; not supported yet')


def receipt(p):
    return dict(protocol_sha256=fingerprint(p),protocol_id=p['protocol_id'],source=p['source'],
        split_definition=p['split'],order=p['order'],sequence_length=p['sequence']['length'],
        sequence=p['sequence'],batch=p['batch'],split_before_order=p['split']['strategy']!='upstream_modulo',
        split_membership_unit='nonempty document',split_seed_used=p['split']['strategy']=='stable_hash',assignment_hash='sha256(seed NUL exact_utf8_text)' if p['split']['strategy']=='stable_hash' else None,
        test_used_for_training=False,decontamination_verified=False,
        limitations=['Hashes do not establish semantic decontamination.',
                     'Only single-worker deterministic partitioning is supported.',
                     'Legacy modulo depends on fixed shuffled stream order.' if p['split']['strategy']=='upstream_modulo' else 'Hash fractions are approximate, not exact counts.'])


def torch_loader(p,role,encoder,epoch=0):
    from torch.utils.data import IterableDataset,DataLoader
    import torch
    class Pairs(IterableDataset):
        def __iter__(self):
            rows=select_rows(source_stream(p,role),p,role,epoch=epoch)
            for x,y in token_pairs(rows,encoder,p):
                yield torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long)
    return DataLoader(Pairs(),batch_size=p['batch']['size'],drop_last=p['batch']['drop_last'],num_workers=0)


def protocol_loaders(args,encoder,epoch=0):
    p=getattr(args,"_data_contract",None) or read(args.data_protocol);validate_runtime(p,args)
    return torch_loader(p,'train',encoder,epoch),torch_loader(p,'validation',encoder)


def fixed_validation_identity(loader,max_batches):
    """Hash only the materialized pairs actually scored; never iterate a loader.

    DataLoader iteration can consume RNG state even without shuffling. Indexing
    the fixed pair dataset avoids introducing an observer-induced random draw.
    """
    batch=loader.batch_size
    if not _integer(batch,1) or not _integer(max_batches,1):raise ValueError('Invalid validation batch limit')
    count=len(loader.dataset)
    if loader.drop_last:count=count//batch*batch
    count=min(count,max_batches*batch)
    if count==0:raise ValueError('No complete validation examples are available')
    digest=hashlib.sha256();targets=0;lengths=set()
    for i in range(count):
        x,y=loader.dataset[i]
        if x.ndim!=1 or y.shape!=x.shape:raise ValueError('Invalid fixed validation pair shape')
        for tensor in (x,y):
            digest.update(tensor.detach().cpu().numpy().astype('<i8',copy=False).tobytes())
        targets+=y.numel();lengths.add(x.numel())
    return dict(sha256=digest.hexdigest(),examples=count,target_tokens=targets,
        batches=(count+batch-1)//batch,sequence_lengths=sorted(lengths),
        serialization='ordered input then target, little-endian int64 bytes',
        scope='materialized fixed validation examples scored under this batch limit',
        semantic_decontamination_verified=False)


class ConsumptionTrace:
    """Hash successful training microbatch inputs, with explicit shape framing."""
    def __init__(self):
        self._digest=hashlib.sha256()
        self.microbatches=0
        self.target_tokens=0

    def observe(self,inputs,targets):
        import struct
        if (inputs.ndim!=2 or inputs.shape!=targets.shape or inputs.numel()==0
            or str(inputs.dtype)!='torch.int64' or str(targets.dtype)!='torch.int64'):
            raise ValueError('Input identity requires nonempty matching int64 [batch,sequence] tensors')
        self._digest.update(struct.pack('<QQ',*inputs.shape))
        for tensor in (inputs,targets):
            raw=tensor.detach().cpu().contiguous().numpy().astype('<i8',copy=False)
            self._digest.update(raw.tobytes())
        self.microbatches+=1
        self.target_tokens+=targets.numel()

    def receipt(self):
        return dict(sha256=self._digest.hexdigest(),microbatches=self.microbatches,
            target_tokens=self.target_tokens,algorithm='sha256-framed-batch-shape-xy-i64le-v1',
            scope='ordered inputs/targets for microbatches with successful backward; not optimizer-state identity',
            semantic_decontamination_verified=False)
