from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from ..domain import RunSpec


@dataclass
class PreparedRun:
    spec: RunSpec
    run_dir: Path
    cwd: Path
    env: dict[str, str]


@dataclass
class ProcessHandle:
    prepared: PreparedRun
    process: subprocess.Popen
    started_monotonic: float
    stdout_handle: TextIO
    stderr_handle: TextIO
    timed_out: bool = False
    stopped: bool = False


@dataclass(frozen=True)
class ProcessObservation:
    running: bool
    exit_code: int | None
    timed_out: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False


class Executor(Protocol):
    def prepare(self, spec: RunSpec, run_dir: str | Path) -> PreparedRun: ...

    def start(self, prepared: PreparedRun) -> ProcessHandle: ...

    def observe(self, handle: ProcessHandle) -> ProcessObservation: ...

    def stop(self, handle: ProcessHandle) -> None: ...

    def collect(self, handle: ProcessHandle, run_dir: str | Path) -> ExecutionResult: ...
