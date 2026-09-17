import sys
import time

from exp_orchestrator.domain import CommandSpec, ResourceRequest, RunSpec
from exp_orchestrator.evidence import EvidenceStore
from exp_orchestrator.executors.native import NativeExecutor
from exp_orchestrator.ledger import Ledger
from exp_orchestrator.scheduler import Scheduler


class DummyAgent:
    def __init__(self, gpus=None):
        self._executor = NativeExecutor()
        self._gpus = list(gpus or [])

    def executor(self, name):
        assert name == "native"
        return self._executor

    def capabilities(self):
        return {
            "host_id": "test-host",
            "gpus": self._gpus,
            "executors": ["native"],
            "cpu_count": 4,
            "memory_total_mb": 8192,
        }


def spec(run_id, *, code="import sys; sys.exit(0)", priority=0, depends_on=None, resources=None):
    return RunSpec(
        run_id=run_id,
        command=CommandSpec(argv=[sys.executable, "-u", "-c", code]),
        priority=priority,
        depends_on=depends_on or [],
        resources=resources or ResourceRequest(),
    )


def make_scheduler(tmp_path, *, gpus=None):
    store = EvidenceStore(tmp_path / "runs")
    ledger = Ledger(tmp_path / "ledger.db", store)
    scheduler = Scheduler(store, ledger, [DummyAgent(gpus=gpus)])
    return scheduler, ledger


def drain(scheduler, ledger, run_ids, limit=200):
    for _ in range(limit):
        scheduler.tick()
        rows = [ledger.get_run(run_id) for run_id in run_ids]
        if all(row and row["status"] in {"completed", "failed", "cancelled", "blocked"} for row in rows):
            return
        time.sleep(0.01)
    raise AssertionError("scheduler did not reach terminal state")


def test_priority_ordering_starts_highest_priority_first(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec("low", code="import time; time.sleep(.1)", priority=1))
    scheduler.enqueue(spec("high", code="import time; time.sleep(.1)", priority=9))
    scheduler.tick()
    assert ledger.get_run("high")["status"] == "running"
    assert ledger.get_run("low")["status"] == "queued"
    drain(scheduler, ledger, ["low", "high"])


def test_dependencies_require_success_and_failure_blocks_dependents(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec("bad", code="import sys; sys.exit(7)", priority=2))
    scheduler.enqueue(spec("child", depends_on=["bad"], priority=1))
    drain(scheduler, ledger, ["bad", "child"])
    assert ledger.get_run("bad")["status"] == "failed"
    assert ledger.get_run("child")["status"] == "blocked"

    scheduler.enqueue(spec("good"))
    scheduler.enqueue(spec("after-good", depends_on=["good"]))
    drain(scheduler, ledger, ["good", "after-good"])
    assert ledger.get_run("after-good")["status"] == "completed"


def test_cancel_retry_and_resource_rejection(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec("long", code="import time; time.sleep(.15)"))
    scheduler.tick()
    assert ledger.get_run("long")["status"] == "running"
    scheduler.cancel("long")
    assert ledger.get_run("long")["status"] == "cancelled"

    retry_id = scheduler.retry("long")
    assert retry_id != "long"
    drain(scheduler, ledger, [retry_id])
    assert ledger.get_run(retry_id)["status"] == "completed"

    impossible = ResourceRequest(gpu_count=1, min_vram_mb=24000)
    scheduler.enqueue(spec("gpu-needed", resources=impossible))
    scheduler.tick()
    assert ledger.get_run("gpu-needed")["status"] == "blocked"


def test_set_priority_changes_queue_order_without_mutating_manifest(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec("first", code="import time; time.sleep(.05)", priority=1))
    scheduler.enqueue(spec("second", code="import time; time.sleep(.05)", priority=2))
    scheduler.set_priority("first", 10)
    scheduler.tick()
    assert ledger.get_run("first")["status"] == "running"
    manifest_priority = scheduler.evidence.load_run("first")["manifest"]["priority"]
    assert manifest_priority == 1
