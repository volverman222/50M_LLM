"""Synthetic boundary calibration only: no training or quality evaluation."""
import sys
from pathlib import Path
import hashlib
import torch
from baseline_runtime import BoundaryTrace, write_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    source, checkpoint, output = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(source/'src'))
    from llm_mini_lab.models import LoopedGPTModel
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = LoopedGPTModel(state['config'])
    model.out_head.weight = model.tok_emb.weight
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    model = model.cuda().eval()
    x = torch.arange(128, device='cuda').repeat(16, 1)
    trace = BoundaryTrace(model, output)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        observed = model(x)
        trace.close()
        reference = model(x)
    assert torch.equal(observed, reference), 'observer changed model outputs'
    summary = dict(kind='synthetic_inference_calibration', trained_result=False,
        checkpoint_sha256=sha(checkpoint), checkpoint_training_tokens=32768,
        input_shape=list(x.shape), input_dtype=str(x.dtype), output_dtype=str(observed.dtype),
        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),
        output_identical=True, optimizer_steps=0,
        interpretation='Shape/dtype/alias evidence only; not a 50M baseline result.')
    write_json(output/'probe_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
