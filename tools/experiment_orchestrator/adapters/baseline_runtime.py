"""Passive representation and parameter-alias inventory for the 50M adapter."""
from pathlib import Path
import json
import torch


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def representation(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'device': str(value.device), 'requires_grad': value.requires_grad,
            'stride': list(value.stride()), 'contiguous': value.is_contiguous()}
    if isinstance(value, (tuple, list)):
        return [representation(v) for v in value]
    return {'python_type': type(value).__name__}


def parameter_inventory(model):
    records = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        row = records.setdefault(id(param), dict(aliases=[], shape=list(param.shape),
            dtype=str(param.dtype), numel=param.numel(), trainable=param.requires_grad))
        row['aliases'].append(name)
    return list(records.values())


class BoundaryTrace:
    """Record exactly one ordinary forward; no tensor mutation or RNG calls."""
    def __init__(self, model, folder):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.model, self.rows, self.handles = model, [], []
        self.done = False
        self.handles.append(model.register_forward_pre_hook(self._begin))
        for name, module in model.named_modules():
            if name:
                self.handles.append(module.register_forward_hook(self._hook(name)))
        self.handles.append(model.register_forward_hook(self._end))

    def _begin(self, module, inputs):
        if not self.done:
            write_json(self.folder/'parameters.json', parameter_inventory(module))
            write_json(self.folder/'buffers.json', [dict(name=n, **representation(b))
                for n, b in module.named_buffers()])

    def _hook(self, name):
        def record(module, inputs, output):
            if not self.done:
                self.rows.append(dict(index=len(self.rows), node=name,
                    module_type=type(module).__name__, inputs=representation(inputs),
                    output=representation(output)))
        return record

    def _end(self, module, inputs, output):
        if not self.done:
            self.rows.append(dict(index=len(self.rows), node='model',
                module_type=type(module).__name__, inputs=representation(inputs),
                output=representation(output)))
            write_json(self.folder/'representation_trace.json', self.rows)
            self.done = True

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
