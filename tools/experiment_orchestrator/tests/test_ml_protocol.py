import copy
import json
import sys
from pathlib import Path
import pytest


def api():
    import importlib.util
    assert importlib.util.find_spec('exp_orchestrator.ml'), 'ML recipe integration missing'
    from exp_orchestrator import ml
    return ml


def recipe(tmp_path):
    import os
    repo=Path(os.environ.get('DEVPOST_MODEL_REPO',str(Path(__file__).parents[3])))
    if not (repo/'src/llm_mini_lab/data_protocol.py').is_file():
        pytest.skip('Set DEVPOST_MODEL_REPO for optional external model integration tests')
    tokenizer=repo/'tokenizers/fineweb_16384_bpe.model'
    from llm_mini_lab.data_protocol import sha256_file
    p=dict(schema_version=1,protocol_id='smoke',
      source=dict(backend='huggingface',dataset='HuggingFaceTB/smollm-corpus',config='cosmopedia-v2',revision='3ba9d605774198c5868892d7a8deda78031a781f',split='train',text_field='text'),
      tokenizer=dict(name='sp16384',sha256=sha256_file(tokenizer)),
      sequence=dict(length=128,packing='packed',add_eos=True,drop_tail=True),
      split=dict(strategy='stable_hash',seed=11,modulus=100,validation_buckets=[0],test_buckets=[1]),
      order=dict(seed=22,shuffle_buffer=100,reseed_each_epoch=True),
      batch=dict(size=2,drop_last=True),num_workers=0)
    return dict(run_id='ml-run',model_repo=str(repo),protocol=p,tokenizer_model=str(tokenizer),
      model=dict(emb_dim=1024,n_heads=8,n_kv_heads=8,n_unique_layers=3,num_loops=2,ff_hidden_dim=1376,positional_encoding='rope'),
      training=dict(target_tokens=100000000,seed=123,gradient_accumulation=4,learning_rate=0.0003,warmup_ratio=0.05,min_lr_ratio=0.1,weight_decay=0.1,grad_clip=1.0,eval_batches=20,eval_every=500),
      execution=dict(device='cpu',timeout_seconds=3600,min_vram_mb=0),depends_on=[])


def test_recipe_compiles_to_identical_embedded_protocol_and_cli(tmp_path):
    ml=api();r=recipe(tmp_path);spec=ml.build_spec(tmp_path,r)
    assert spec.metadata['data_protocol']==r['protocol']
    options=dict(zip(spec.command.argv[3::2],spec.command.argv[4::2]))
    assert options['--seed']=='123' and options['--context-length']=='128'
    assert '--data-protocol' in spec.command.argv and '--hf-upload-every' in spec.command.argv
    assert spec.metadata['data_protocol']['order']['seed']==22
    assert spec.metadata['data_protocol']['split']['seed']==11
    assert spec.command.env['WANDB_MODE']=='disabled'
    assert spec.resources.gpu_count==0


def test_protocol_artifact_is_sealed_and_visible_after_submit(tmp_path):
    ml=api();r=recipe(tmp_path)
    from exp_orchestrator.operations import submit,read_tail
    spec=ml.build_spec(tmp_path,r);submit(tmp_path,spec.model_dump(mode='json'))
    artifact=tmp_path/'runs/ml-run/artifacts/data_protocol.json'
    assert json.loads(artifact.read_text())==r['protocol']
    assert 'stable_hash' in read_tail(tmp_path,'ml-run','data-protocol')
    # Editing submitted protocol is an evidence failure, not a new experiment.
    artifact.write_text('{}')
    from exp_orchestrator.operations import run_worker
    with pytest.raises(ValueError,match='protocol'):run_worker(tmp_path,interval=0.01)
    assert not (tmp_path/'runs/ml-run/result.json').exists()


def test_validation_and_test_source_profiles_cannot_be_silently_turned_into_training(tmp_path):
    ml=api();r=recipe(tmp_path);r['protocol']['source']['split']='validation'
    with pytest.raises(ValueError,match='training source'):ml.build_spec(tmp_path,r)


def test_cli_exposes_serve_watch_prepare_without_network(tmp_path):
    api()
    from exp_orchestrator.cli import parser
    for args in [['serve','--port','0'],['watch','--once'],['prepare-ml','recipe.json']]:
        parsed=parser().parse_args(['--root',str(tmp_path),*args])
        assert parsed.action==args[0]


@pytest.mark.parametrize('device', ['auto', '', 'remote'])
def test_orchestrated_device_is_explicit_not_unreserved_auto(tmp_path,device):
    ml=api();r=recipe(tmp_path);r['execution']['device']=device
    with pytest.raises(ValueError,match='device'):ml.build_spec(tmp_path,r)


@pytest.mark.parametrize('key,value', [('learning_rate',float('inf')),('weight_decay',float('nan')),('grad_clip',float('inf'))])
def test_training_numbers_are_finite(tmp_path,key,value):
    ml=api();r=recipe(tmp_path);r['training'][key]=value
    with pytest.raises(ValueError,match='finite'):ml.build_spec(tmp_path,r)


def test_model_protocol_dry_run_is_cpu_offline_and_does_not_create_checkpoints(tmp_path):
    ml=api();r=recipe(tmp_path)
    r['model'].update(emb_dim=16,n_heads=2,n_kv_heads=1,n_unique_layers=1,num_loops=2,ff_hidden_dim=32)
    spec=ml.build_spec(tmp_path,r)
    import subprocess, os
    protocol_path=tmp_path/'protocol.json'
    protocol_path.write_text(json.dumps(r['protocol']))
    command=list(spec.command.argv)
    command[command.index('--data-protocol')+1]=str(protocol_path)
    command.append('--dry-run')
    result=subprocess.run(command,cwd=r['model_repo'],env=dict(os.environ,**spec.command.env,
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert not (tmp_path/'runs/ml-run/checkpoints').exists()


def test_real_tiny_cpu_training_flows_through_queue_with_local_named_splits(tmp_path):
    """Synthetic wiring test, not a project quality/training result."""
    ml=api();r=recipe(tmp_path)
    from llm_mini_lab.data_protocol import sha256_file,fingerprint
    files={}
    for role in ('train','validation','test'):
        path=tmp_path/(role+'.jsonl')
        prefix={'train':'Training fixture','validation':'Validation fixture','test':'HELDOUT NEVER TRAIN'}[role]
        path.write_text(''.join(json.dumps({'text':f'{prefix} {i}: a synthetic sentence for software testing only.'})+'\n' for i in range(30)),encoding='utf-8')
        files[role]=str(path)
    hashes={k:sha256_file(v) for k,v in files.items()}
    r['protocol']['source']=dict(backend='local_jsonl',files=files,hashes=hashes,text_field='text')
    r['protocol']['split']=dict(strategy='source_splits',names={k:k for k in files})
    r['protocol']['sequence']['length']=8
    r['protocol']['order']['shuffle_buffer']=8
    r['model'].update(emb_dim=16,n_heads=2,n_kv_heads=1,n_unique_layers=1,num_loops=2,ff_hidden_dim=32)
    r['training'].update(target_tokens=256,gradient_accumulation=1,eval_batches=2,eval_every=8,warmup_ratio=0.1)
    from exp_orchestrator.operations import submit,run_worker,snapshot
    spec=ml.build_spec(tmp_path/'evidence',r)
    payload=spec.model_dump(mode='json')
    payload['command']['env'].update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    submit(tmp_path/'evidence',payload)
    assert run_worker(tmp_path/'evidence',interval=0.01,resource_interval=1)==0
    folder=tmp_path/'evidence/runs/ml-run'
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='completed',(folder/'stderr.log').read_text()
    assert result['metrics']['tokens_seen']==256
    receipt=json.loads((folder/'checkpoints/data_protocol_receipt.json').read_text())
    assert receipt['protocol_sha256']==fingerprint(r['protocol'])
    assert receipt['sequence_length']==8 and receipt['test_used_for_training'] is False
    fixed=json.loads((folder/"checkpoints/fixed_validation_identity.json").read_text())
    assert fixed["examples"]==4 and fixed["target_tokens"]==32
    assert len(fixed["sha256"])==64
    summary=json.loads((folder/'checkpoints/training_summary.json').read_text())
    assert summary['dataset']=='local_jsonl'
    used=json.loads((folder/'checkpoints/training_input_identity.json').read_text())
    assert used['target_tokens']==256 and used['microbatches']==16
    assert used['protocol_sha256']==fingerprint(r['protocol'])
    assert summary['training_input_sha256']==used['sha256']
    assert spec.metadata['source_provenance']['files']

    assert all(sha256_file(v)==hashes[k] for k,v in files.items())
    assert snapshot(tmp_path/'evidence')[0]['status']=='completed'


def test_prepare_has_no_runtime_files_and_records_selected_model_interpreter(tmp_path):
    ml=api();r=recipe(tmp_path);r['execution']['python_executable']=sys.executable
    evidence=tmp_path/'not_created'
    spec=ml.build_spec(evidence,r)
    assert spec.command.argv[0]==str(Path(sys.executable).resolve())
    assert spec.metadata['model_python']==spec.command.argv[0]
    assert not evidence.exists()
    r['execution']['python_executable']=str(tmp_path/'missing-python')
    with pytest.raises(ValueError,match='interpreter'):ml.build_spec(evidence,r)
