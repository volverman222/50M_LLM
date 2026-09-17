from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import psutil

from .base import ExecutionResult, PreparedRun, ProcessHandle, ProcessObservation
from ..domain import RunSpec


class NativeExecutor:
    name = "native"

    def prepare(self, spec: RunSpec, run_dir: str | Path) -> PreparedRun:
        if 'source_provenance' in spec.metadata:
            from ..provenance import verify
            verify(spec.metadata['source_provenance'])
        run_path = Path(run_dir)
        cwd = Path(spec.command.cwd) if spec.command.cwd else run_path
        env = os.environ.copy()
        env.update(spec.command.env)
        return PreparedRun(spec=spec, run_dir=run_path, cwd=cwd, env=env)

    def start(self, prepared: PreparedRun) -> ProcessHandle:
        prepared.run_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = (prepared.run_dir / "stdout.log").open("a", encoding="utf-8", buffering=1)
        stderr_handle = (prepared.run_dir / "stderr.log").open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            prepared.spec.command.argv,
            cwd=prepared.cwd,
            env=prepared.env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
        )
        return ProcessHandle(
            prepared=prepared,
            process=process,
            started_monotonic=time.monotonic(),
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    def observe(self, handle: ProcessHandle) -> ProcessObservation:
        exit_code = handle.process.poll()
        if exit_code is None and handle.prepared.spec.timeout_seconds is not None:
            elapsed = time.monotonic() - handle.started_monotonic
            if elapsed >= handle.prepared.spec.timeout_seconds:
                handle.timed_out = True
                self.stop(handle)
                exit_code = handle.process.poll()
        return ProcessObservation(
            running=exit_code is None,
            exit_code=exit_code,
            timed_out=handle.timed_out,
        )

    def stop(self, handle: ProcessHandle) -> None:
        if handle.process.poll() is not None:
            handle.stopped = True
            return
        try:
            parent = psutil.Process(handle.process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs(children + [parent], timeout=2)
            for proc in alive:
                proc.kill()
        except psutil.Error:
            handle.process.terminate()
        try:
            handle.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=3)
        handle.stopped = True

    def collect(self, handle: ProcessHandle, run_dir: str | Path) -> ExecutionResult:
        if handle.process.poll() is None:
            handle.process.wait()
        handle.stdout_handle.flush()
        handle.stderr_handle.flush()
        handle.stdout_handle.close()
        handle.stderr_handle.close()
        return ExecutionResult(
            exit_code=handle.process.returncode,
            timed_out=handle.timed_out,
            cancelled=handle.stopped and not handle.timed_out,
        )
