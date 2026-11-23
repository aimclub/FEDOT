from copy import deepcopy
import os
from pathlib import Path
import pytest
from examples.advanced.diagnostics.fallback_rng_contract import analytic_score, compare, run_case


def test_scalar_is_graph_dependent_and_key_order_independent():
    a=[dict(operation='logit',params={'C':1},parents=[])]
    b=[dict(parents=[],params={'C':1},operation='logit')]
    assert analytic_score(a)==analytic_score(b)
    b[0]['params']['C']=2
    assert analytic_score(a)!=analytic_score(b)


def test_differential_gate_rejects_flag_specific_trials():
    records={}
    for stream in ['a','b']:
        for flag in ['false','true']:
            records[f'{stream}-{flag}']=dict(calls=[stream]*6,entropy=[],fmin=[],returned_graph=[],history=[],fits=0)
    assert compare(records)['passed']
    records['a-true']['calls'][1]='damage'
    with pytest.raises(AssertionError):compare(records)


@pytest.mark.skipif(not os.environ.get('FEDOT_H24_STAGE'),reason='frozen real composer/Hyperopt host')
def test_actual_ineligible_dispatch_common_entropy_without_fits(tmp_path):
    stage=Path(os.environ['FEDOT_H24_STAGE'])
    graph=[dict(operation='logit',params={'C':1.0},parents=[1]),dict(operation='scaling',params={},parents=[])]
    results=[]
    for enabled in [False,True]:
        out=tmp_path/str(enabled);out.mkdir()
        results.append(run_case(stage,out,enabled,104754,deepcopy(graph)))
    for key in ['calls','entropy','fmin','returned_graph','history']:
        assert results[0][key]==results[1][key]
