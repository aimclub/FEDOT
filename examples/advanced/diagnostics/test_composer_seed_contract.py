import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from examples.advanced.diagnostics.composer_seed_patch import parse_seed


@pytest.mark.parametrize('value,expected',[(None,None),(0,0),(104758,104758),('104758',104758),('0',0)])
def test_seed_forms(value,expected):assert parse_seed(value)==expected


@pytest.mark.parametrize('value',[True,False,-1,1.5,'-1','1.0','abc',' 1','١'])
def test_invalid_seed(value):
    with pytest.raises(ValueError):parse_seed(value)


@pytest.mark.skipif(not os.environ.get('FEDOT_H26_STAGE'),reason='isolated composer host')
def test_real_factory_validation_precedes_constructor():
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    adapter=load_adapter(Path(os.environ['FEDOT_H26_STAGE']))
    config=SimpleNamespace(framework_params={'_tuner_random_state':True})
    with patch.object(adapter,'Fedot',side_effect=AssertionError('Constructor must not run')) as factory:
        with pytest.raises(ValueError):adapter._make_fedot(config,'neg_log_loss',{},54,5.4)
        factory.assert_not_called()


@pytest.mark.skipif(not os.environ.get('FEDOT_H26_STAGE'),reason='isolated composer host')
def test_real_seeded_dispatch_both_routes(tmp_path):
    from examples.advanced.diagnostics.composer_seed_contract import run_case
    stage=Path(os.environ['FEDOT_H26_STAGE'])
    graph=[dict(operation='logit',params={'C':1.},parents=[1]),dict(operation='scaling',params={},parents=[])]
    for name in ['original-seeded','paid-seeded']:
        out=tmp_path/name;out.mkdir();r=run_case(stage,out,name,graph,smoke=True)
        assert r['seed']==104758 and r['unseeded_requests']==0 and r['fits']==0
