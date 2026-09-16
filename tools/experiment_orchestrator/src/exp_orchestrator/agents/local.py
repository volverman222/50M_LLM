from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any

import psutil

from ..executors.base import Executor


class LocalHostAgent:
    def __init__(self, executors: dict[str, Executor]):
        self._executors = dict(executors)
        self._host_id = platform.node() or os.environ.get("COMPUTERNAME", "localhost")

    def executor(self, name: str) -> Executor:
        try:
            return self._executors[name]
        except KeyError as exc:
            raise KeyError(f"executor not registered: {name}") from exc

    def _gpu_inventory(self) -> list[dict[str, Any]]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        except (OSError, subprocess.SubprocessError):
            return []
        rows: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                rows.append({
                    "index": int(parts[0]), "name": parts[1],
                    "memory_total_mb": int(parts[2]), "memory_free_mb": int(parts[3]),
                })
        return rows

    def capabilities(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        return {
            "host_id": self._host_id,
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True) or 1,
            "memory_total_mb": int(memory.total / (1024 * 1024)),
            "memory_available_mb": int(memory.available / (1024 * 1024)),
            "gpus": self._gpu_inventory(),
            "executors": sorted(self._executors),
        }
