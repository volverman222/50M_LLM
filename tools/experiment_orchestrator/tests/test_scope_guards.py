import json
import sys
import pytest
from exp_orchestrator.domain import CommandSpec, RunSpec
from exp_orchestrator.evidence import EvidenceStore
from exp_orchestrator.ledger import Ledger


def test_run_id_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path).create_run(RunSpec(run_id='../escape',
            command=CommandSpec(argv=[sys.executable])))


def test_manifest_edit_is_detected(tmp_path):
    store = EvidenceStore(tmp_path)
    folder = store.create_run(RunSpec(run_id='check',
        command=CommandSpec(argv=[sys.executable])))
    obj = json.loads((folder / 'manifest.json').read_text())
    obj['metadata']['changed'] = True
    (folder / 'manifest.json').write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='digest'):
        store.load_run('check')


def test_artifact_cannot_escape_run(tmp_path):
    store = EvidenceStore(tmp_path/'runs')
    store.create_run(RunSpec(run_id='check', command=CommandSpec(argv=['x'])))
    src = tmp_path/'fixture.txt'; src.write_text('data')
    with pytest.raises(ValueError):
        store.register_artifact('check', src, '../../escape.txt')
