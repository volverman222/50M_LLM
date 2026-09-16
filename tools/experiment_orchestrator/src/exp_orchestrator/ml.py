"""Compile a reviewed explicit ML recipe; never launch or upload here."""
from pathlib import Path
import sys
import math
from .domain import RunSpec,CommandSpec,ResourceRequest
from .evidence import safe_run_dir
from .bridge import load_protocol, resolve_recipe_paths
from .provenance import capture


def build_spec(root,recipe):
    api=load_protocol(recipe.get('model_repo',''))
    validate,_keys,fingerprint,validate_runtime=api.validate,api._keys,api.fingerprint,api.validate_runtime
    from types import SimpleNamespace
    _keys(recipe,['run_id','model_repo','protocol','tokenizer_model','model','training','execution','depends_on'])
    protocol=validate(recipe['protocol']);model=recipe['model'];training=recipe['training'];execution=recipe['execution']
    _keys(model,['emb_dim','n_heads','n_kv_heads','n_unique_layers','num_loops','ff_hidden_dim','positional_encoding'])
    _keys(training,['target_tokens','seed','gradient_accumulation','learning_rate','warmup_ratio','min_lr_ratio','weight_decay','grad_clip','eval_batches','eval_every'])
    _keys(execution,['device','timeout_seconds','min_vram_mb'],['python_executable'])
    source=protocol['source'];split=protocol['split']
    train_split=split['names']['train'] if split['strategy']=='source_splits' else source.get('split','train')
    if train_split.lower() in ('validation','valid','val','test'):
        raise ValueError('A held-out native split cannot be the training source')
    for key in ['emb_dim','n_heads','n_kv_heads','n_unique_layers','num_loops','ff_hidden_dim']:
        if type(model[key]) is not int or model[key]<=0:raise ValueError('Invalid model dimensions')
    d,f,h,kv=model['emb_dim'],model['ff_hidden_dim'],model['n_heads'],model['n_kv_heads']
    if d%h or h%kv or model['positional_encoding'] not in ('rope','learned'):raise ValueError('Invalid attention/position configuration')
    vocab=16384 if protocol['tokenizer']['name']=='sp16384' else 50257
    estimate=vocab*d+model['n_unique_layers']*(2*d*d+2*d*(d//h*kv)+3*d*f+2*f+6*d)+2*d
    if model['positional_encoding']=='learned':estimate+=protocol['sequence']['length']*d
    if estimate>50000000:raise ValueError(f'Configured looped model exceeds 50M parameters: {estimate}')
    for key in ['target_tokens','gradient_accumulation','eval_batches','eval_every']:
        if type(training[key]) is not int or training[key]<=0:raise ValueError('Training counts must be positive integers')
    if type(training['seed']) is not int or training['seed']<0:raise ValueError('Initialization seed must be a nonnegative integer')
    for key in ['learning_rate','warmup_ratio','min_lr_ratio','weight_decay','grad_clip']:
        if isinstance(training[key],bool) or not isinstance(training[key],(int,float)) or not math.isfinite(training[key]):
            raise ValueError('Training numbers must be finite real values')
    if not 0<training['warmup_ratio']<1 or not 0<=training['min_lr_ratio']<=1:raise ValueError('Invalid learning-rate schedule')
    if not training['learning_rate']>0 or not training['grad_clip']>0 or not training['weight_decay']>=0:raise ValueError('Invalid optimizer settings')
    if execution['device'] not in ('cpu','cuda','mps'):raise ValueError('Use an explicit device: cpu, cuda or mps; auto cannot reserve resources safely')
    repo=Path(recipe['model_repo']).resolve();tokenizer=Path(recipe['tokenizer_model']).resolve() if recipe['tokenizer_model'] else None
    validate_runtime(protocol,SimpleNamespace(context_length=protocol['sequence']['length'],micro_batch_size=protocol['batch']['size'],num_workers=protocol['num_workers'],tokenizer=protocol['tokenizer']['name'],tokenizer_model=tokenizer,resume=None))
    folder=safe_run_dir(Path(root).resolve()/'runs',recipe['run_id'])
    python=Path(execution.get('python_executable') or sys.executable).resolve()
    if not python.is_file():raise ValueError('Selected model Python interpreter is missing')
    script=repo/'scripts/train_pretrain_1b.py'
    if not script.is_file():raise ValueError('Model training script is absent')
    args={'data-protocol':str(folder/'artifacts/data_protocol.json'),
        'context-length':protocol['sequence']['length'],'micro-batch-size':protocol['batch']['size'],
        'num-workers':0,'tokenizer':protocol['tokenizer']['name'],
        **{key.replace('_','-'):val for key,val in model.items()},
        **{key.replace('_','-'):val for key,val in training.items()},
        'device':execution['device'],'checkpoint-dir':str(folder/'checkpoints'),
        'telemetry-jsonl':str(folder/'telemetry/training.jsonl'),'telemetry-every':25,
        'hf-upload-every':0,'benchmark-every':0,'hourly-cost':0,'run-name':recipe['run_id']}
    if tokenizer:args['tokenizer-model']=str(tokenizer)
    argv=[str(python),'-u',str(script)]
    for key,val in args.items():argv.extend(['--'+key,str(val)])
    return RunSpec(run_id=recipe['run_id'],profile='ml-configured',depends_on=recipe['depends_on'],
        command=CommandSpec(argv=argv,cwd=str(repo),env={'PYTHONPATH':str(repo/'src'),'PYTHONUTF8':'1',
            'PYTHONUNBUFFERED':'1','WANDB_MODE':'disabled','PYTHONHASHSEED':str(training['seed'])}),
        resources=ResourceRequest(gpu_count=1 if execution['device']=='cuda' else 0,
            min_vram_mb=execution['min_vram_mb'],min_disk_free_mb=4096),
        timeout_seconds=execution['timeout_seconds'],max_retries=0,
        metadata={'project':'Devpost Hackathon — 50M_LLM','data_protocol':protocol,
            'source_provenance':capture(repo),'data_protocol_sha256':fingerprint(protocol),'initialization_seed':training['seed'],'model_python':str(python),
            'estimated_parameter_count':estimate,'parameter_count_limit':50000000,
            'completion':{'summary':'checkpoints/training_summary.json','min_tokens':training['target_tokens']},
            'evaluation_scope':'local validation only; official harness and WikiText-103 are separate jobs'})
