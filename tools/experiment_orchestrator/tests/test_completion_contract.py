"""False-success guards for training completion evidence."""
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from exp_orchestrator.scheduler import Scheduler


def verdict(tmp_path,rule,data):
    folder=tmp_path/'run';folder.mkdir()
    (tmp_path/'outside.json').write_text(json.dumps(data))
    (folder/'summary.json').write_text(json.dumps(data))
    handle=SimpleNamespace(prepared=SimpleNamespace(run_dir=folder,spec=SimpleNamespace(metadata={'completion':rule})))
    result=SimpleNamespace(cancelled=False,timed_out=False,exit_code=0)
    return Scheduler(None,None,[])._completion('run',handle,result)[0]


def test_completion_summary_cannot_escape_run(tmp_path):
    assert verdict(tmp_path,{'summary':'../outside.json','min_tokens':100},{'target_reached':True,'tokens_seen':100})=='failed'


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1])
def test_nonfinite_or_invalid_token_count_cannot_complete(tmp_path,value):
    assert verdict(tmp_path,{'summary':'summary.json','min_tokens':100},{'target_reached':True,'tokens_seen':value})=='failed'


def test_target_reached_must_be_boolean_true(tmp_path):
    assert verdict(tmp_path,{'summary':'summary.json','min_tokens':100},{'target_reached':'false','tokens_seen':100})=='failed'
