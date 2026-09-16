"""Pinned, local-only baseline launcher; source/model behavior is not replaced."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
from baseline_runtime import BoundaryTrace, write_json


def main():
    folder = Path(sys.argv[1]).resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    meta = manifest['metadata']; source = Path.cwd()
    args = meta['training_argv']
    if '--wandb' in args or args[args.index('--hf-upload-every')+1] != '0':
        raise ValueError('baseline external publishing is disabled')
    if args[args.index('--target-tokens')+1] != '50000000':
        raise ValueError('only the approved nominal 50M-token budget is admitted')
    def phase(name):
        item = dict(phase=name, process='trainer', at=datetime.now(timezone.utc).isoformat())
        with (folder/'telemetry/phases.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(item)+'\n')
        print('PHASE', name, flush=True)
    phase('adapter_start')
    sys.path[:0] = [str(source/'src'), str(source/'scripts')]
    import torch
    import datasets
    import train_pretrain_1b as trainer
    original_load = datasets.load_dataset
    def pinned_load(name, *a, **kw):
        if name != meta['dataset']:
            raise ValueError('unapproved dataset')
        kw['revision'] = meta['dataset_revision']
        return original_load(name, *a, **kw)
    datasets.load_dataset = pinned_load
    objects = {}
    original_model = trainer.LoopedGPTModel
    def observed_model(cfg):
        if cfg != meta['model']:
            raise ValueError('resolved model configuration differs from manifest')
        model = original_model(cfg)
        objects['model'] = model
        objects['trace'] = BoundaryTrace(model, folder/'characterization')
        return model
    trainer.LoopedGPTModel = observed_model
    original_fixed = trainer.make_fixed_eval_loaders
    def fixed_loaders(*a, **kw):
        phase('data_preparing')
        train, valid = original_fixed(*a, **kw)
        objects['validation'] = valid
        pairs = list(valid.dataset)
        h = hashlib.sha256()
        for x, y in pairs:
            h.update(x.numpy().tobytes()); h.update(y.numpy().tobytes())
        torch.save(pairs, folder/'artifacts/fixed_validation.pt')
        write_json(folder/'metrics/validation_identity.json', dict(
            sha256=h.hexdigest(), pairs=len(pairs), split='held_out_document_modulo',
            dataset=meta['dataset'], revision=meta['dataset_revision'],
            tokenizer_sha256=hashlib.sha256((source/'tokenizers/fineweb_16384_bpe.model').read_bytes()).hexdigest()))
        phase('data_ready')
        return train, valid
    trainer.make_fixed_eval_loaders = fixed_loaders
    sys.argv = [str(source/'scripts/train_pretrain_smollm_1b.py'), *args]
    try:
        trainer.main(dataset='smollm')
        phase('training_returned')
        model = objects['model']
        loss = trainer.evaluate(model, objects['validation'], 20,
            next(model.parameters()).device, True, torch.bfloat16)
        write_json(folder/'metrics/final_validation.json', dict(
            loss=loss, unit='nats_per_token', source='fixed_validation', final=True))
        phase('final_validation_complete')
    except Exception as exc:
        phase('failed_'+type(exc).__name__)
        raise
    finally:
        if 'trace' in objects:
            objects['trace'].close()


if __name__ == '__main__':
    main()
