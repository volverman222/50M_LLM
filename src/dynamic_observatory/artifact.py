from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "devpost.dynamic_observation.v1"
ALLOWED_SOURCES = frozenset({"LIVE", "REPLAY", "DERIVED", "N/A"})


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class DynamicObservation:
    run_id: str
    checkpoint_tokens: int
    probe_id: str
    trace_name: str
    source: str
    states: list[list[float]]
    metrics: Mapping[str, Any]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _required_text(self.run_id, "run_id")
        _required_text(self.probe_id, "probe_id")
        _required_text(self.trace_name, "trace_name")
        if self.schema != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(f"source must be one of {sorted(ALLOWED_SOURCES)}")
        if isinstance(self.checkpoint_tokens, bool) or not isinstance(self.checkpoint_tokens, int):
            raise ValueError("checkpoint_tokens must be an integer")
        if self.checkpoint_tokens < 0:
            raise ValueError("checkpoint_tokens must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "checkpoint_tokens": self.checkpoint_tokens,
            "probe_id": self.probe_id,
            "trace_name": self.trace_name,
            "source": self.source,
            "states": [list(row) for row in self.states],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicObservation":
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        states = payload.get("states")
        metrics = payload.get("metrics")
        if not isinstance(states, list) or not isinstance(metrics, Mapping):
            raise ValueError("states must be a list and metrics must be an object")
        normalized_states: list[list[float]] = []
        for row in states:
            if not isinstance(row, list):
                raise ValueError("states rows must be lists")
            normalized_states.append([float(value) for value in row])
        return cls(
            run_id=str(payload.get("run_id", "")),
            checkpoint_tokens=payload.get("checkpoint_tokens"),
            probe_id=str(payload.get("probe_id", "")),
            trace_name=str(payload.get("trace_name", "")),
            source=str(payload.get("source", "")),
            states=normalized_states,
            metrics=dict(metrics),
        )


def write_observation(path: Path, observation: DynamicObservation) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(observation.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load_observation(path: Path) -> DynamicObservation:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observation document must be an object")
    return DynamicObservation.from_dict(payload)
