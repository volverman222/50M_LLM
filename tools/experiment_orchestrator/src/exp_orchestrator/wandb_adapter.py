"""Explicit scalar export bridge. Training evidence and scheduling stay external."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .domain import CommandSpec


def build_wandb_export_command(
    run_dir: Path | str,
    model_repo: Path | str,
    output_dir: Path | str,
    *,
    project: str,
    entity: str | None = None,
    mode: str = "offline",
    allow_online: bool = False,
    api_key_env: str | None = None,
    python: str = sys.executable,
) -> CommandSpec:
    """Build, never execute, an export of a successfully completed run's JSONL."""
    run_dir, model_repo, output_dir = map(
        lambda p: Path(p).resolve(), (run_dir, model_repo, output_dir)
    )
    if mode not in {"offline", "online"} or (mode == "online" and not allow_online):
        raise ValueError("Online export requires explicit authorization")
    if not project.strip():
        raise ValueError("A W&B project is required")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Export outputs must be outside sealed run evidence")
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise ValueError("Export requires a terminal completed run")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise ValueError("Export requires a successful completed run")
    telemetry = run_dir / "telemetry/training.jsonl"
    script = model_repo / "scripts/export_telemetry_wandb.py"
    if not telemetry.is_file() or not script.is_file():
        raise ValueError("Telemetry or compatible model-repository exporter is missing")
    argv = [
        python,
        "-u",
        str(script),
        str(telemetry),
        "--project",
        project,
        "--name",
        run_dir.name,
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
    ]
    if entity:
        argv += ["--entity", entity]
    if mode == "online":
        argv += ["--allow-online"]
    if api_key_env:
        if not api_key_env.isidentifier():
            raise ValueError("Supply an environment variable name, never its value")
        argv += ["--api-key-env", api_key_env]
    return CommandSpec(
        argv=argv,
        cwd=str(model_repo),
        env={
            "PYTHONPATH": str(model_repo / "src"),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_MODE": mode,
        },
    )
