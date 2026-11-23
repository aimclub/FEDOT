import os
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from examples.advanced.diagnostics.live_seed_contract import settings,synthetic,redirect_child,applicability,trace_tuner


def test_only_flag_differs_and_evolution_is_unrestricted():
    a=settings(Path('/run'),False);b=settings(Path('/run'),True)
    assert a.pop('_paid_allrow_tuning') is False and b.pop('_paid_allrow_tuning') is True
    assert a==b and a['_tuner_random_state']==104760
    assert a['num_of_generations']==1 and a['pop_size']==4
    assert not {'available_operations','initial_assumption','num_trees','_predefined_model'} & a.keys()
    x,y=synthetic();assert x.shape==(120,6) and np.array_equal(np.bincount(y[:90]),[30,30,30])


def test_redirect_preserves_arguments_and_interpreter():
    real=Mock(return_value='ok');stage=Path('/stage');redirect=redirect_child(real,stage)
    assert redirect('caller',str(stage/'live_adapter_contract.py'),'child',python_exec='python',input_data={})=='ok'
    real.assert_called_once_with('caller',str(stage/'live_seed_contract.py'),'child',python_exec='python',input_data={})
    with pytest.raises(AssertionError):redirect('caller','wrong')


@pytest.mark.parametrize('kind',['graph','policy','partial'])
def test_inapplicable_is_not_a_neutrality_pass(kind):
    a,b='graph','graph';policy={'status':'ineligible_original_path'}
    calls=[[{'valid':True} for _ in range(6)] for _ in range(2)]
    assert applicability(a,b,policy,calls) is None
    if kind=='graph':b='other'
    elif kind=='policy':policy={'status':'evaluated'}
    else:calls[1][-1]['valid']=False
    assert applicability(a,b,policy,calls) is not None


@pytest.mark.skipif(not os.environ.get('FEDOT_H29_STAGE'),reason='actual frozen composer/Hyperopt observer')
def test_trace_forwards_real_seeded_fmin_unchanged(tmp_path):
    from examples.advanced.diagnostics.composer_seed_contract import run_case
    graph=[dict(operation='logit',params={'C':1.},parents=[1]),dict(operation='scaling',params={},parents=[])]
    record=dict(builders=[],fmin=[])
    with trace_tuner(record):
        result=run_case(Path(os.environ['FEDOT_H29_STAGE']),tmp_path,'original-seeded',graph,smoke=True)
    assert result['fits']==0 and len(result['calls'])==6
    assert record['builders']==[dict(seed=104758,generator=True)] and record['same_rng']
    assert len(record['fmin'])==2 and all(c['succeeded'] for c in record['fmin'])
    assert record['fmin'][0]['after']==record['fmin'][1]['before']
