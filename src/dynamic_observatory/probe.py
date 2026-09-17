from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .artifact import DynamicObservation
from .capture import TraceRecorder
from .metrics import analyze_trajectory


def discover_repeated_modules(
    model: nn.Module,
    *model_args: Any,
    min_calls: int = 2,
) -> list[str]:
    """Find outermost named modules invoked repeatedly by one inference pass."""
    counts: dict[str, int] = defaultdict(int)
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(lambda _m, _i, _o, n=name: counts.__setitem__(n, counts[n] + 1)))

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(*model_args)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    repeated = sorted((name for name, count in counts.items() if count >= min_calls), key=lambda n: (n.count("."), n))
    selected: list[str] = []
    for name in repeated:
        if any(name.startswith(parent + ".") for parent in selected):
            continue
        selected.append(name)
    return selected


def capture_observations(
    model: nn.Module,
    probe_inputs: torch.Tensor,
    *,
    run_id: str,
    checkpoint_tokens: int,
    probe_id: str,
    module_names: Sequence[str] | None = None,
    projection_bases: Mapping[str, torch.Tensor] | None = None,
    max_period: int = 8,
) -> list[DynamicObservation]:
    names = list(module_names) if module_names is not None else discover_repeated_modules(model, probe_inputs)
    if not names:
        return []

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode(), TraceRecorder(model, names) as recorder:
            model(probe_inputs)
    finally:
        model.train(was_training)

    records: list[DynamicObservation] = []
    bases = projection_bases or {}
    for name in names:
        trajectory = recorder.trajectory(name)
        basis = bases.get(name)
        report = analyze_trajectory(trajectory, projection_basis=basis, max_period=max_period)
        metrics = report.metrics_dict()
        metrics.update({
            "steps": int(trajectory.shape[0]),
            "state_dim": int(trajectory.shape[1]),
            "projection_basis": "FIXED_REFERENCE" if basis is not None else "LOCAL_PCA_UNCALIBRATED",
        })
        records.append(DynamicObservation(
            run_id=run_id,
            checkpoint_tokens=checkpoint_tokens,
            probe_id=probe_id,
            trace_name=name,
            source="DERIVED",
            states=trajectory.tolist(),
            metrics=metrics,
        ))
    return records
