from pathlib import Path
import numpy as np
import pytest
from examples.advanced.diagnostics.live_adapter_contract import synthetic, settings, check_predictions


def test_fixed_synthetic_data_and_bounded_unrestricted_evolution():
    x,y=synthetic();xx,yy=synthetic()
    assert x.shape==(120,6) and np.array_equal(x,xx) and np.array_equal(y,yy)
    assert np.array_equal(np.bincount(y[:90]),[30,30,30])
    a=settings(Path('/run'),False);b=settings(Path('/run'),True)
    assert a.pop('_paid_allrow_tuning') is False and b.pop('_paid_allrow_tuning') is True
    assert a==b and a['num_of_generations']==1 and a['pop_size']==4
    assert not {'available_operations','initial_assumption','_predefined_model','num_trees'} & a.keys()
    assert not any(k.startswith('use_') and k.endswith('_cache') for k in a)


@pytest.mark.parametrize('damage',['truth','nan','range','normalization','classes','predictions'])
def test_prediction_audit_rejects_corruption(damage):
    _,y=synthetic();truth=y[90:].copy();probs=np.eye(3)[truth]*.7+.1
    predictions=truth.copy()
    check_predictions(probs,predictions,truth)
    if damage=='truth':truth[0]=2
    elif damage=='nan':probs[0,0]=np.nan
    elif damage=='range':probs[0,0]=-1
    elif damage=='normalization':probs[0]*=.5
    elif damage=='classes':probs=probs[:,:2]
    elif damage=='predictions':predictions[0]=2
    with pytest.raises(AssertionError):check_predictions(probs,predictions,truth)
