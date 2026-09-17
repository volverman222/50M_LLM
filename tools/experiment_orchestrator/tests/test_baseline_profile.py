import sys
from pathlib import Path
import pytest
from exp_orchestrator.baseline import APPROVED, build_baseline_spec


def test_only_approved_recipe_builds_local_launch(tmp_path):
    spec = build_baseline_spec('baseline-test', tmp_path/'source', tmp_path/'runs',
        sys.executable, 'a'*40)
    args = spec.metadata['training_argv']
    assert args[args.index('--target-tokens')+1] == '50000000'
    assert args[args.index('--hf-upload-every')+1] == '0'
    assert '--wandb' not in args
    assert spec.metadata['model'] == APPROVED
    assert spec.max_retries == 0
    assert spec.command.env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] == '1'


def test_rejects_configuration_drift(tmp_path):
    with pytest.raises(ValueError, match='approved'):
        build_baseline_spec('bad', tmp_path, tmp_path, sys.executable,
            'a'*40, model={**APPROVED, 'num_loops': 3})


def test_missing_data_revision_is_not_reproducible(tmp_path):
    with pytest.raises(ValueError, match='revision'):
        build_baseline_spec('bad', tmp_path, tmp_path, sys.executable, '')
