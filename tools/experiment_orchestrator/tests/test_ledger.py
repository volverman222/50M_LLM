import sqlite3
from contextlib import closing

from exp_orchestrator.domain import CommandSpec, RunEvent, RunResult, RunSpec, RunState
from exp_orchestrator.evidence import EvidenceStore
from exp_orchestrator.ledger import Ledger


def spec(run_id: str, priority: int = 0) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        campaign_id="campaign-a",
        profile="generic",
        command=CommandSpec(argv=["python", "job.py"]),
        priority=priority,
    )


def snapshot(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        runs = conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        metrics = conn.execute("SELECT * FROM metrics ORDER BY run_id, name").fetchall()
        artifacts = conn.execute("SELECT * FROM artifacts ORDER BY run_id, path").fetchall()
    return runs, metrics, artifacts


def test_project_query_and_rebuild_are_deterministic(tmp_path):
    evidence_root = tmp_path / "runs"
    db_path = tmp_path / "ledger.db"
    store = EvidenceStore(evidence_root)
    ledger = Ledger(db_path, store)

    store.create_run(spec("run-a", priority=2))
    store.append_event("run-a", RunEvent.transition("queued", "running", actor="scheduler"))
    ledger.project_run("run-a")

    store.create_run(spec("run-b", priority=5))
    store.append_event("run-b", RunEvent.transition("queued", "running", actor="scheduler"))
    store.seal_result("run-b", RunResult(
        run_id="run-b", status=RunState.COMPLETED, exit_code=0, metrics={"loss": 2.5}
    ))
    artifact = tmp_path / "metric.txt"
    artifact.write_text("payload", encoding="utf-8")
    store.register_artifact("run-b", artifact, "reports/metric.txt")
    ledger.project_run("run-b")

    before = snapshot(db_path)
    assert ledger.get_run("run-a")["status"] == "running"
    assert ledger.get_run("run-b")["status"] == "completed"
    assert [row["run_id"] for row in ledger.query_runs({"campaign_id": "campaign-a"})] == ["run-b", "run-a"]

    db_path.unlink()
    ledger = Ledger(db_path, store)
    ledger.rebuild(evidence_root)
    assert snapshot(db_path) == before
