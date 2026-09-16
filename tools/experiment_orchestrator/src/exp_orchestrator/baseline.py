"""Launch adapter restricted to the single user-approved diagnostic baseline."""
from pathlib import Path
import re
from .domain import CommandSpec, ResourceRequest, RunSpec

APPROVED = dict(vocab_size=16384, context_length=128, emb_dim=1024,
    n_heads=8, n_unique_layers=3, num_loops=2, drop_rate=0.0,
    qkv_bias=False, positional_encoding='rope', ff_activation='swiglu',
    ff_hidden_dim=1376, tokenizer_name='sp16384')


def build_baseline_spec(run_id, source, runs_root, python, revision, *, model=None):
    if model is not None and model != APPROVED:
        raise ValueError('configuration differs from the approved baseline')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('dataset revision must be a resolved commit SHA')
    source, folder = Path(source).resolve(), (Path(runs_root)/run_id).resolve()
    options = dict(target_tokens=50_000_000, context_length=128, tokenizer='sp16384',
        tokenizer_model=str(source/'tokenizers/fineweb_16384_bpe.model'),
        emb_dim=1024, n_heads=8, n_unique_layers=3, num_loops=2,
        positional_encoding='rope', ff_hidden_dim=1376, micro_batch_size=16,
        gradient_accumulation=4, learning_rate=0.0003, weight_decay=0.1,
        warmup_ratio=0.05, min_lr_ratio=0.1, grad_clip=1.0, seed=123,
        num_workers=0, shuffle_buffer=10000, val_mod=100,
        smollm_config='cosmopedia-v2', eval_every=250, eval_batches=20,
        log_every=10, save_every=1000, hf_upload_every=0, benchmark_every=0)
    options.update(checkpoint_dir=str(folder/'checkpoints'),
        telemetry_jsonl=str(folder/'telemetry/training.jsonl'), telemetry_every=25,
        device='cuda', hourly_cost=0, run_name=run_id)
    argv = [value for key, val in options.items()
        for value in ('--'+key.replace('_', '-'), str(val))]
    env = dict(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(source/'src'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='disabled', OTEL_SDK_DISABLED='true')
    return RunSpec(run_id=run_id, campaign_id='approved-baseline-50m',
        profile='ml-baseline', command=CommandSpec(
            argv=[str(python), '-u', str(source/'baseline_entry.py'), str(folder)],
            cwd=str(source), env=env),
        resources=ResourceRequest(gpu_count=1, min_vram_mb=4096, min_disk_free_mb=4096),
        metadata={'model': dict(APPROVED), 'training_argv': argv,
            'dataset': 'HuggingFaceTB/smollm-corpus', 'dataset_config': 'cosmopedia-v2',
            'dataset_revision': revision, 'execution_authorized': True,
            'completion': {'summary': 'checkpoints/training_summary.json',
                           'min_tokens': 50_000_000},
            'budget_note': 'Nominal budget; upstream final microbatch may round up.',
            'cost_note': 'Hourly cost unset; zero is not an energy-cost measurement.',
            'architecture_changes_authorized': False,
            'upload_authorized': False, 'automatic_retry_authorized': False})
