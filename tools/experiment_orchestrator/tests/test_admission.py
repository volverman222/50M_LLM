import sys
import pytest
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



from exp_orchestrator.domain import ResourceRequest, RunSpec


def test_disk_headroom_is_an_admission_gate(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    s = spec('disk-block', resources=ResourceRequest(min_disk_free_mb=10**12))
    scheduler.enqueue(s); scheduler.tick()
    assert ledger.get_run('disk-block')['status'] == 'blocked'
    assert 'disk' in scheduler.evidence.load_run('disk-block')['result']['message']
    assert scheduler.active == {}


def test_sweep_previews_cannot_be_submitted(tmp_path):
    scheduler, _ = make_scheduler(tmp_path)
    data = spec('preview').model_dump()
    data['metadata'] = {'preview_only': True}
    with pytest.raises(ValueError, match='preview'):
        scheduler.enqueue(RunSpec.model_validate(data))


def test_restart_does_not_duplicate_unknown_running_process(tmp_path):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec('unknown')); scheduler.enqueue(spec('wait'))
    scheduler._transition('unknown', 'running', {'pid': 999999})
    scheduler.tick()
    assert ledger.get_run('wait')['status'] == 'queued'


def test_terminal_receipt_must_persist_before_completion_projection(tmp_path, monkeypatch):
    scheduler, ledger = make_scheduler(tmp_path)
    scheduler.enqueue(spec('terminal-write-failure'))
    def disk_full(*args):
        raise OSError('disk full')
    monkeypatch.setattr(scheduler.evidence, 'seal_result', disk_full)
    with pytest.raises(OSError):
        scheduler._finish('terminal-write-failure', 'completed', 0)
    assert ledger.get_run('terminal-write-failure')['status'] == 'queued'
