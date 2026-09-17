"""Standalone invariants: source identity, no overwrite, explicit model bridge."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest


def test_explicit_model_repo_is_used_instead_of_installed_model(tmp_path):
    from exp_orchestrator.ml import build_spec
    root=tmp_path/'chosen_model'
    path=root/'src/llm_mini_lab/data_protocol.py';path.parent.mkdir(parents=True)
    path.write_text("raise ValueError('SELECTED_MODEL_PROTOCOL')",encoding='utf-8')
    with pytest.raises(ValueError,match='SELECTED_MODEL_PROTOCOL'):
        build_spec(tmp_path/'evidence',{'model_repo':str(root)})
    assert not (tmp_path/'evidence').exists()


def test_missing_explicit_model_protocol_is_not_replaced_by_global_install(tmp_path):
    from exp_orchestrator.ml import build_spec
    with pytest.raises(ValueError,match='protocol'):
        build_spec(tmp_path/'evidence',{'model_repo':str(tmp_path/'missing')})
    assert not (tmp_path/'evidence').exists()


def snapshot_api():
    assert importlib.util.find_spec('exp_orchestrator.distribution'), 'Snapshot API missing'
    from exp_orchestrator import distribution
    return distribution


def package_fixture(tmp_path):
    src=tmp_path/'package';(src/'src/exp_orchestrator').mkdir(parents=True)
    (src/'pyproject.toml').write_text('[project]\nname="devpost-experiment-orchestrator"\nversion="0.3.0"\n')
    (src/'README.md').write_text('Devpost Hackathon')
    (src/'src/exp_orchestrator/__init__.py').write_text('VERSION=1')
    (src/'.env').write_text('DO_NOT_COPY=1')
    (src/'runtime').mkdir();(src/'runtime/private.json').write_text('{}')
    return src


def test_snapshot_copies_only_source_and_verifies_complete_tree(tmp_path):
    api=snapshot_api();src=package_fixture(tmp_path);dest=tmp_path/'release'
    receipt=api.export_package(src,dest)
    assert receipt['files']['src/exp_orchestrator/__init__.py']
    assert not (dest/'.env').exists() and not (dest/'runtime').exists()
    assert api.verify_package(dest)['files']==receipt['files']
    assert api.compare_package(src,dest)['matches'] is True
    (dest/'src/exp_orchestrator/__init__.py').write_text('VERSION=2')
    with pytest.raises(ValueError,match='identity'):api.verify_package(dest)


def test_snapshot_refuses_existing_destination_and_nested_export(tmp_path):
    api=snapshot_api();src=package_fixture(tmp_path);dest=tmp_path/'release'
    dest.mkdir();(dest/'keep').write_text('preserved')
    with pytest.raises(FileExistsError):api.export_package(src,dest)
    assert (dest/'keep').read_text()=='preserved'
    with pytest.raises(ValueError):api.export_package(src,src/'release')


def test_unrecorded_importable_file_invalidates_release(tmp_path):
    api=snapshot_api();src=package_fixture(tmp_path);dest=tmp_path/'release'
    api.export_package(src,dest)
    (dest/'src/exp_orchestrator/extra.py').write_text('pass')
    with pytest.raises(ValueError,match='identity'):api.verify_package(dest)


def test_relative_recipe_paths_resolve_against_recipe_not_current_directory(tmp_path):
    from exp_orchestrator import ml
    assert hasattr(ml,'resolve_recipe_paths'), 'Portable recipe resolution missing'
    original={'model_repo':'../model','tokenizer_model':'tokenizer.model',
        'protocol':{'source':{'backend':'local_jsonl','files':{'train':'train.jsonl'}}},
        'execution':{'python_executable':sys.executable}}
    resolved=ml.resolve_recipe_paths(original,tmp_path)
    assert resolved['model_repo']==str((tmp_path/'../model').resolve())
    assert resolved['protocol']['source']['files']['train']==str(tmp_path/'train.jsonl')
    assert original['model_repo']=='../model'


def test_standalone_cli_exposes_export_and_verification():
    from exp_orchestrator.cli import parser
    for args in [['snapshot-package','source','destination'],['verify-package','source']]:
        parsed=parser().parse_args(['--root','unused',*args])
        assert parsed.action==args[0]
