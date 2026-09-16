"""Defensive protocol checks; not claims about existing model pathology."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from llm_mini_lab import data_protocol as dp


def config():
    root=Path(__file__).parents[1]
    return dp.read(root/'configs/data_protocols/smollm_cosmopedia_hash_128.json')


@pytest.mark.parametrize('strategy', ['stable_hash','source_splits'])
@pytest.mark.parametrize('heldout', ['test','validation','val','valid','dev'])
def test_direct_protocol_rejects_known_heldout_training_sources(strategy,heldout):
    p=config()
    if strategy=='source_splits':
        p['split']={'strategy':strategy,'names':{'train':heldout,'validation':'heldout_validation','test':None}}
    else:p['source']['split']=heldout
    with pytest.raises(ValueError,match='held-out'):
        dp.validate(p)


def test_schema_version_boolean_is_not_integer_one():
    p=config();p['schema_version']=True
    with pytest.raises(ValueError):dp.validate(p)


def test_training_consumption_identity_is_exact_and_rng_neutral():
    assert hasattr(dp, 'ConsumptionTrace'), 'Actual input identity is missing'
    x=torch.arange(16,dtype=torch.int64).reshape(2,8);y=x+1
    before=torch.get_rng_state().clone()
    a=dp.ConsumptionTrace();b=dp.ConsumptionTrace();c=dp.ConsumptionTrace()
    for trace in (a,b):trace.observe(x,y);trace.observe(x+2,y+2)
    c.observe(x+2,y+2);c.observe(x,y)
    assert a.receipt()==b.receipt()
    assert a.receipt()['sha256']!=c.receipt()['sha256']
    assert a.receipt()['target_tokens']==32 and a.receipt()['microbatches']==2
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(x,torch.arange(16).reshape(2,8))


def test_consumption_trace_rejects_invalid_or_empty_inputs():
    assert hasattr(dp, 'ConsumptionTrace'), 'Actual input identity is missing'
    trace=dp.ConsumptionTrace()
    for x,y in [(torch.ones(2,4),torch.ones(2,4)),
                (torch.zeros(0,4,dtype=torch.long),torch.zeros(0,4,dtype=torch.long)),
                (torch.ones(2,4,dtype=torch.long),torch.ones(2,3,dtype=torch.long))]:
        with pytest.raises(ValueError):trace.observe(x,y)
    assert trace.receipt()['microbatches']==0
