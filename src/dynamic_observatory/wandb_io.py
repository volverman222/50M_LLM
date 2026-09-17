from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from .artifact import DynamicObservation, write_observation


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "trace"


def _scalar_metrics(observation: DynamicObservation) -> dict[str, float]:
    prefix = f"dynamics/{_safe(observation.trace_name)}"
    out: dict[str, float] = {}
    for key, value in observation.metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[f"{prefix}/{key}"] = float(value)
        elif key == "closure_errors" and isinstance(value, dict):
            for period, error in value.items():
                if isinstance(error, (int, float)):
                    out[f"{prefix}/closure_q{period}"] = float(error)
    return out


def log_observations_to_wandb(
    run,
    observations: Iterable[DynamicObservation],
    output_dir: Path,
    *,
    update: int,
) -> list[Path]:
    import wandb

    observations = list(observations)
    if not observations:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Artifact(
        name=f"dynamic-observations-{_safe(str(run.id))}-{observations[0].checkpoint_tokens}",
        type="dynamic-observation",
        metadata={
            "schema": observations[0].schema,
            "probe_id": observations[0].probe_id,
            "checkpoint_tokens": observations[0].checkpoint_tokens,
        },
    )
    paths: list[Path] = []
    scalars: dict[str, float | int] = {"update": update}
    for observation in observations:
        path = output_dir / f"{_safe(observation.trace_name)}.json"
        write_observation(path, observation)
        artifact.add_file(str(path), name=path.name)
        paths.append(path)
        scalars.update(_scalar_metrics(observation))
    run.log(scalars)
    run.log_artifact(artifact)
    return paths
