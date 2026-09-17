from pathlib import Path
import pytest
from exp_orchestrator.domain import RunSpec,CommandSpec
from exp_orchestrator.executors.native import NativeExecutor


def test_changed_source_is_rejected_before_a_process_starts(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('exp_orchestrator.provenance'), 'Source identity gate missing'
    from exp_orchestrator.provenance import capture
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True);(repo/'src/pkg').mkdir(parents=True)
    script=repo/'scripts/train_pretrain_1b.py';script.write_text('print(123)')
    module=repo/'src/pkg/__init__.py';module.write_text('VERSION=1')
    identity=capture(repo)
    spec=RunSpec(run_id='test',command=CommandSpec(argv=['python',str(script)],cwd=str(repo)),
        metadata={'source_provenance':identity})
    assert NativeExecutor().prepare(spec,tmp_path/'output').cwd==repo
    module.write_text('VERSION=2')
    with pytest.raises(ValueError,match='source'):NativeExecutor().prepare(spec,tmp_path/'output')
    assert not (tmp_path/'output/stdout.log').exists()


def test_new_importable_source_file_is_a_provenance_change(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('exp_orchestrator.provenance'), 'Source identity gate missing'
    from exp_orchestrator.provenance import capture,verify
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True);(repo/'src').mkdir()
    (repo/'scripts/train_pretrain_1b.py').write_text('pass')
    identity=capture(repo);(repo/'src/unrecorded.py').write_text('pass')
    with pytest.raises(ValueError,match='source'):verify(identity)
