from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import fit_projection_basis

SCHEMA = "devpost.dynamic_reference_frame.v1"


def _finite_vector(values: list[float], field: str) -> list[float]:
    out = [float(value) for value in values]
    if not out or not all(math.isfinite(value) for value in out):
        raise ValueError(f"{field} must contain finite values")
    return out


@dataclass(frozen=True)
class DynamicReferenceFrame:
    run_id: str
    checkpoint_tokens: int
    probe_id: str
    trace_name: str
    center: list[float]
    basis: list[list[float]]
    source: str = "DERIVED"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        if self.source != "DERIVED":
            raise ValueError("reference frames must be DERIVED")
        if not self.run_id or not self.probe_id or not self.trace_name:
            raise ValueError("run_id, probe_id, and trace_name are required")
        if (
            isinstance(self.checkpoint_tokens, bool)
            or not isinstance(self.checkpoint_tokens, int)
            or self.checkpoint_tokens < 0
        ):
            raise ValueError("checkpoint_tokens must be a non-negative integer")
        center = _finite_vector(self.center, "center")
        if not self.basis or len(self.basis) != len(center):
            raise ValueError("basis must have one row per state dimension")
        width = len(self.basis[0])
        if width < 1 or width > 3:
            raise ValueError("basis must contain 1 to 3 projection components")
        for row in self.basis:
            if len(row) != width:
                raise ValueError("basis rows must have equal width")
            _finite_vector(row, "basis[]")

    @property
    def state_dim(self) -> int:
        return len(self.center)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "run_id": self.run_id,
            "checkpoint_tokens": self.checkpoint_tokens,
            "probe_id": self.probe_id,
            "trace_name": self.trace_name,
            "state_dim": self.state_dim,
            "center": list(self.center),
            "basis": [list(row) for row in self.basis],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicReferenceFrame":
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        center = payload.get("center")
        basis = payload.get("basis")
        if not isinstance(center, list) or not isinstance(basis, list):
            raise ValueError("center and basis must be lists")
        frame = cls(
            run_id=str(payload.get("run_id", "")),
            checkpoint_tokens=payload.get("checkpoint_tokens"),
            probe_id=str(payload.get("probe_id", "")),
            trace_name=str(payload.get("trace_name", "")),
            center=[float(value) for value in center],
            basis=[[float(value) for value in row] for row in basis],
            source=str(payload.get("source", "")),
        )
        state_dim = payload.get("state_dim")
        if state_dim is not None and state_dim != frame.state_dim:
            raise ValueError("state_dim does not match center length")
        return frame


def build_reference_frame(
    states: torch.Tensor,
    *,
    probe_id: str,
    trace_name: str,
    run_id: str,
    checkpoint_tokens: int,
    components: int = 3,
) -> DynamicReferenceFrame:
    states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must have shape [steps, dimensions]")
    center = states.mean(dim=0)
    basis = fit_projection_basis(states, components=components)
    return DynamicReferenceFrame(
        run_id=run_id,
        checkpoint_tokens=checkpoint_tokens,
        probe_id=probe_id,
        trace_name=trace_name,
        center=center.tolist(),
        basis=basis.tolist(),
    )


def project_states(states: torch.Tensor, frame: DynamicReferenceFrame) -> torch.Tensor:
    states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
    if states.ndim != 2 or states.shape[1] != frame.state_dim:
        raise ValueError("states do not match reference frame dimensionality")
    center = torch.tensor(frame.center, dtype=torch.float32)
    basis = torch.tensor(frame.basis, dtype=torch.float32)
    return (states - center) @ basis


def write_reference_frame(path: Path, frame: DynamicReferenceFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frame.as_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_reference_frame(path: Path) -> DynamicReferenceFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference frame document must be an object")
    return DynamicReferenceFrame.from_dict(payload)
