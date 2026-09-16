import json
import sys
from exp_orchestrator.domain import RunSpec, CommandSpec


def test_cli_dummy_run_logs_and_ledger_rebuild(tmp_path, capsys):
    from exp_orchestrator.cli import main
    spec = RunSpec(run_id='dummy', command=CommandSpec(argv=[
        sys.executable, '-u', '-c', 'print("verified dummy")']))
    source = tmp_path/'dummy.json'; source.write_text(spec.model_dump_json())
    prefix = ['--root', str(tmp_path/'store')]
    assert main(prefix+['submit', str(source)]) == 0
    assert main(prefix+['worker']) == 0
    folder = tmp_path/'store/runs/dummy'
    assert json.loads((folder/'result.json').read_text())['status'] == 'completed'
    assert 'verified dummy' in (folder/'stdout.log').read_text()
    assert main(prefix+['logs', 'dummy']) == 0
    assert main(prefix+['rebuild']) == 0
    assert main(prefix+['show', 'dummy']) == 0
    assert 'completed' in capsys.readouterr().out


def test_worker_lock_has_one_owner(tmp_path):
    import pytest
    from exp_orchestrator.locking import worker_lock
    with worker_lock(tmp_path):
        with pytest.raises(OSError):
            with worker_lock(tmp_path):
                pass
