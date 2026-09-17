import hashlib
import json
from pathlib import Path

import pytest

from exp_orchestrator.domain import CommandSpec, RunEvent, RunResult, RunSpec, RunState
from exp_orchestrator.evidence import EvidenceStore


def make_spec(run_id: str = "run-001") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        profile="generic",
        command=CommandSpec(argv=["python", "job.py"]),
    )


def test_create_run_builds_canonical_layout_and_manifest_is_immutable(tmp_path):
    store = EvidenceStore(tmp_path)
    run_dir = store.create_run(make_spec())
    assert json.loads((run_dir / "manifest.json").read_text())["run_id"] == "run-001"
    for filename in ("source.json", "environment.json", "hardware.json", "events.jsonl", "stdout.log", "stderr.log"):
        assert (run_dir / filename).exists()
    for dirname in ("telemetry", "metrics", "artifacts", "checkpoints"):
        assert (run_dir / dirname).is_dir()
    with pytest.raises(FileExistsError):
        store.create_run(make_spec())


def test_events_and_logs_append_in_order(tmp_path):
    store = EvidenceStore(tmp_path)
    store.create_run(make_spec())
    first = RunEvent.transition("queued", "admitted", actor="scheduler")
    second = RunEvent.transition("admitted", "running", actor="local")
    store.append_event("run-001", first)
    store.append_event("run-001", second)
    store.append_log("run-001", "stdout", "hello\n")
    store.append_log("run-001", "stderr", "warn\n")
    events = [json.loads(line) for line in (tmp_path / "run-001" / "events.jsonl").read_text().splitlines()]
    assert [row["to_state"] for row in events] == ["admitted", "running"]
    assert (tmp_path / "run-001" / "stdout.log").read_text() == "hello\n"
    assert (tmp_path / "run-001" / "stderr.log").read_text() == "warn\n"


def test_artifact_registration_hashes_copy_and_result_seals_once(tmp_path):
    store = EvidenceStore(tmp_path)
    run_dir = store.create_run(make_spec())
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abc123")
    record = store.register_artifact("run-001", source, "model/payload.bin")
    expected = hashlib.sha256(b"abc123").hexdigest()
    assert record["sha256"] == expected
    assert (run_dir / "artifacts" / "model" / "payload.bin").read_bytes() == b"abc123"

    result = RunResult(run_id="run-001", status=RunState.COMPLETED, exit_code=0, metrics={"loss": 1.25})
    store.seal_result("run-001", result)
    assert json.loads((run_dir / "result.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        store.seal_result("run-001", result)


def test_load_run_returns_manifest_events_and_optional_result(tmp_path):
    store = EvidenceStore(tmp_path)
    store.create_run(make_spec())
    store.append_event("run-001", RunEvent.transition("queued", "admitted", actor="scheduler"))
    loaded = store.load_run("run-001")
    assert loaded["manifest"]["run_id"] == "run-001"
    assert loaded["events"][0]["to_state"] == "admitted"
    assert loaded["result"] is None
