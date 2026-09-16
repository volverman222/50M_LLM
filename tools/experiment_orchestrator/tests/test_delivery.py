import importlib.util
import json
import sys
import time
from pathlib import Path
import pytest


def api():
    assert importlib.util.find_spec('exp_orchestrator'), 'Portable orchestrator package is missing'
    from exp_orchestrator import operations
    return operations


def dummy(run_id='demo', text='hello'):
    return dict(run_id=run_id, command=dict(argv=[sys.executable, '-u', '-c', f'print({text!r})']), timeout_seconds=10)


def test_portable_end_to_end_worker_tail_rebuild(tmp_path):
    m = api()
    m.submit(tmp_path, dummy())
    assert m.run_worker(tmp_path, interval=0.01, resource_interval=0.02) == 0
    rows = m.snapshot(tmp_path)
    assert rows[0]['status'] == 'completed'
    assert 'hello' in m.read_tail(tmp_path, 'demo', 'stdout')
    assert json.loads((tmp_path/'runs/demo/environment.json').read_text())['python_version']
    assert (tmp_path/'runs/demo/telemetry/resources.jsonl').is_file()
    from exp_orchestrator.ledger import Ledger
    from exp_orchestrator.evidence import EvidenceStore
    (tmp_path/'ledger.db').unlink()
    Ledger(tmp_path/'ledger.db', EvidenceStore(tmp_path/'runs')).rebuild()
    assert m.snapshot(tmp_path)[0]['status'] == 'completed'


def test_ui_reads_are_bounded_and_do_not_mutate_evidence(tmp_path):
    m = api()
    m.submit(tmp_path, dummy())
    p = tmp_path/'runs/demo/stdout.log'
    p.write_text('x'*100000+'end')
    assert len(m.read_tail(tmp_path, 'demo', 'stdout', limit=4096).encode()) <= 4096
    assert m.read_tail(tmp_path, 'demo', 'stdout', limit=4096).endswith('end')
    with pytest.raises(ValueError): m.read_tail(tmp_path, '../demo', 'stdout')
    with pytest.raises(ValueError): m.read_tail(tmp_path, 'demo', '../manifest.json')
    assert m.snapshot(tmp_path)[0]['status'] == 'queued'


def test_missing_or_tampered_seal_fails_closed(tmp_path):
    m = api()
    m.submit(tmp_path, dummy())
    seal = tmp_path/'runs/demo/manifest.sha256'
    seal.unlink()
    with pytest.raises(ValueError): m.run_worker(tmp_path, interval=0.01)


def test_cancellation_is_request_until_worker_confirms(tmp_path):
    m = api()
    m.submit(tmp_path, dummy())
    m.request_cancel(tmp_path, 'demo')
    assert m.snapshot(tmp_path)[0]['status'] == 'queued'
    m.run_worker(tmp_path, interval=0.01)
    assert m.snapshot(tmp_path)[0]['status'] == 'cancelled'
    assert m.read_tail(tmp_path, 'demo', 'stdout') == ''


def test_submission_copies_nested_inputs_and_rejects_credentials(tmp_path):
    m = api()
    spec = dummy(); m.submit(tmp_path, spec)
    spec['command']['argv'][-1] = 'raise Exception()'
    assert 'hello' in json.loads((tmp_path/'runs/demo/manifest.json').read_text())['command']['argv'][-1]
    secret = dummy('secret'); secret['command']['env'] = {'WANDB_API_KEY': 'do-not-persist'}
    with pytest.raises(ValueError): m.submit(tmp_path, secret)
    assert not (tmp_path/'runs/secret').exists()


def test_terminal_renderer_includes_status_and_literal_text(tmp_path):
    m = api()
    m.submit(tmp_path, dummy())
    from exp_orchestrator.monitor import terminal_view
    from rich.console import Console
    from io import StringIO
    out = StringIO(); console = Console(file=out, width=140, force_terminal=False)
    console.print(terminal_view(tmp_path, 'demo', 'stdout'))
    assert 'demo' in out.getvalue() and 'queued' in out.getvalue()


def test_safe_run_ids_reject_windows_devices(tmp_path):
    m = api()
    for name in ('CON', 'nul.txt', 'COM1', 'a.', '..'):
        with pytest.raises(ValueError): m.submit(tmp_path, dummy(name))


def test_publish_package_does_not_require_torch():
    import tomllib
    cfg = tomllib.loads((Path(__file__).parents[1]/'pyproject.toml').read_text())
    assert not any('torch' in d for d in cfg['project']['dependencies'])
    assert cfg['project']['scripts']['expctl'] == 'exp_orchestrator.cli:main'
