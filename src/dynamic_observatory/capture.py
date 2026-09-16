from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _reduce_state(tensor: torch.Tensor) -> torch.Tensor:
    state = tensor.detach().to(dtype=torch.float32, device="cpu")
    if state.ndim == 0:
        return state.reshape(1)
    if state.ndim == 1:
        return state.clone()
    dims = tuple(range(state.ndim - 1))
    return state.mean(dim=dims)


class TraceRecorder:
    """Passively records reduced outputs from selected named modules."""

    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        *,
        max_events_per_module: int = 256,
    ):
        self.model = model
        self.module_names = tuple(module_names)
        self.max_events_per_module = max_events_per_module
        self._events: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[Any] = []

    def __enter__(self) -> "TraceRecorder":
        modules = dict(self.model.named_modules())
        missing = [name for name in self.module_names if name not in modules]
        if missing:
            raise KeyError(f"unknown module names: {missing}")
        for name in self.module_names:
            self._handles.append(modules[name].register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            events = self._events[name]
            if len(events) >= self.max_events_per_module:
                return
            tensor = _first_tensor(output)
            if tensor is not None:
                events.append(_reduce_state(tensor))

        return capture

    def trajectory(self, name: str) -> torch.Tensor:
        events = self._events.get(name, [])
        if not events:
            raise KeyError(f"no captured events for module {name!r}")
        return torch.stack(events, dim=0)

    def event_counts(self) -> dict[str, int]:
        return {name: len(events) for name, events in self._events.items()}
