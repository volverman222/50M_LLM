"""Opt-in structural telemetry; sampled reductions may synchronize the device."""

from __future__ import annotations

import torch
from .characterization import tensor_record, gate_record, rms


def _rms(tensor: torch.Tensor) -> float:
    value = tensor.detach().float()
    return torch.sqrt(torch.mean(value * value)).item()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left = a.detach().float().reshape(-1)
    right = b.detach().float().reshape(-1)
    value = torch.nn.functional.cosine_similarity(left, right, dim=0, eps=1e-12)
    return float(value.clamp(-1.0, 1.0).item())


class LoopTelemetryCollector:
    """Observe recurrent block reuse without changing model outputs."""

    def __init__(self, model):
        self.model = model
        self.blocks = list(model.trf_blocks)
        if not self.blocks:
            raise ValueError("model.trf_blocks must contain at least one block")
        self.enabled = False
        self._handles = []
        self._block_records = []
        self._loop_records = []
        self._active = None
        self._loop_entry = None
        for block_index, block in enumerate(self.blocks):
            self._handles.append(
                block.register_forward_pre_hook(self._make_pre(block_index))
            )
            self._handles.append(block.att.register_forward_hook(self._attention_hook))
            self._handles.append(block.ff.register_forward_hook(self._mlp_hook))
            self._handles.append(
                block.register_forward_hook(self._make_post(block_index))
            )
            ff = block.ff.layers
            if hasattr(ff, "gate_proj"):
                self._handles.append(
                    ff.gate_proj.register_forward_hook(self._gate_hook)
                )
                self._handles.append(
                    ff.value_proj.register_forward_hook(self._value_hook)
                )
                self._handles.append(
                    ff.out_proj.register_forward_pre_hook(self._product_hook)
                )

    def start_step(self) -> None:
        self._block_records = []
        self._loop_records = []
        self._active = None
        self._loop_entry = None
        self.enabled = True

    def stop_step(self) -> None:
        self.enabled = False
        self._active = None
        self._loop_entry = None

    def block_records(self) -> list[dict[str, float | int]]:
        return list(self._block_records)

    def loop_records(self) -> list[dict[str, float | int]]:
        return list(self._loop_records)

    def _make_pre(self, block_index):
        def hook(_module, inputs):
            if not self.enabled:
                return
            call_index = len(self._block_records)
            loop_index = call_index // len(self.blocks)
            x = inputs[0]
            self._active = {
                "loop": loop_index,
                "block": block_index,
                "input": x.detach(),
                "attention_rms": float("nan"),
                "mlp_rms": float("nan"),
            }
            if block_index == 0:
                self._loop_entry = x.detach()

        return hook

    def _attention_hook(self, _module, _inputs, output):
        if self.enabled and self._active is not None:
            self._active["attention_rms"] = _rms(output)

    def _mlp_hook(self, _module, _inputs, output):
        if self.enabled and self._active is not None:
            self._active["mlp_rms"] = _rms(output)

    def _make_post(self, block_index):
        def hook(_module, _inputs, output):
            if not self.enabled or self._active is None:
                return
            x_in = self._active.pop("input")
            row = dict(self._active)
            row.update(tensor_record(output))
            row.update(
                {
                    "hidden_rms": _rms(output),
                    "delta_rms": _rms(output - x_in),
                    "cosine_in_out": _cosine(x_in, output),
                }
            )
            self._block_records.append(row)
            if block_index == len(self.blocks) - 1 and self._loop_entry is not None:
                self._loop_records.append(
                    {
                        "loop": row["loop"],
                        "loop_entry_rms": _rms(self._loop_entry),
                        "loop_exit_rms": _rms(output),
                        "loop_delta_rms": _rms(output - self._loop_entry),
                        "loop_cosine": _cosine(self._loop_entry, output),
                    }
                )
                self._loop_entry = None
            self._active = None

        return hook

    def _gate_hook(self, _module, _inputs, output):
        if self.enabled and self._active is not None:
            self._active.update(gate_record(output))

    def _value_hook(self, _module, _inputs, output):
        if self.enabled and self._active is not None:
            self._active["ff_value_rms"] = rms(output)

    def _product_hook(self, _module, inputs):
        if self.enabled and self._active is not None:
            self._active["ff_product_rms"] = rms(inputs[0])

    def close(self) -> None:
        self.stop_step()
        for handle in self._handles:
            handle.remove()
        self._handles = []


def parameter_rms(module: torch.nn.Module) -> float:
    total_sq = 0.0
    total_count = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float()
        total_sq += value.square().sum().item()
        total_count += value.numel()
    return (total_sq / max(total_count, 1)) ** 0.5


def gradient_rms(module: torch.nn.Module) -> float:
    total_sq = 0.0
    total_count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        total_sq += value.square().sum().item()
        total_count += value.numel()
    return (total_sq / max(total_count, 1)) ** 0.5


def snapshot_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [
        parameter.detach().float().cpu().clone()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def parameter_update_rms(
    module: torch.nn.Module,
    before: list[torch.Tensor],
) -> float:
    current = [p for p in module.parameters() if p.requires_grad]
    if len(current) != len(before):
        raise ValueError("parameter snapshot does not match module")
    total_sq = 0.0
    total_count = 0
    for parameter, previous in zip(current, before):
        value = parameter.detach().float().cpu()
        if value.shape != previous.shape:
            raise ValueError("parameter snapshot shape mismatch")
        delta = value - previous
        total_sq += delta.square().sum().item()
        total_count += delta.numel()
    return (total_sq / max(total_count, 1)) ** 0.5
