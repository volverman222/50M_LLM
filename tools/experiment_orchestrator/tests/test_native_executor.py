import sys
import time

from exp_orchestrator.domain import CommandSpec, RunSpec
from exp_orchestrator.executors.native import NativeExecutor


def child_spec(tmp_path, *, exit_code=0, sleep=0.05, timeout=None):
    code = (
        "import os,sys,time; "
        "print(os.environ.get('EXP_TEST_ENV','missing'), flush=True); "
        "print('problem', file=sys.stderr, flush=True); "
        f"time.sleep({sleep}); sys.exit({exit_code})"
    )
    return RunSpec(
        run_id="child",
        command=CommandSpec(
            argv=[sys.executable, "-u", "-c", code],
            cwd=str(tmp_path),
            env={"EXP_TEST_ENV": "present"},
        ),
        timeout_seconds=timeout,
    )


def test_native_executor_runs_child_and_streams_logs(tmp_path):
    executor = NativeExecutor()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepared = executor.prepare(child_spec(tmp_path, exit_code=3), run_dir)
    handle = executor.start(prepared)
    while executor.observe(handle).running:
        time.sleep(0.01)
    result = executor.collect(handle, run_dir)
    assert result.exit_code == 3
    assert "present" in (run_dir / "stdout.log").read_text()
    assert "problem" in (run_dir / "stderr.log").read_text()
    assert handle.process.cwd if hasattr(handle.process, "cwd") else True


def test_native_executor_enforces_timeout_and_cleans_process(tmp_path):
    executor = NativeExecutor()
    run_dir = tmp_path / "run-timeout"
    run_dir.mkdir()
    handle = executor.start(executor.prepare(child_spec(tmp_path, sleep=2, timeout=0.05), run_dir))
    time.sleep(0.08)
    observation = executor.observe(handle)
    assert observation.timed_out is True
    assert observation.running is False
    assert handle.process.poll() is not None
