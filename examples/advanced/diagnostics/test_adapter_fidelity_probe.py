import numpy as np
import pytest
from examples.advanced.diagnostics.adapter_fidelity_probe import compare_arrays


def test_rounding_is_separate_from_exact_equality():
    a=np.array([1.234567890123],dtype='float64')
    r=compare_arrays(a,a.astype('float32'))
    assert not r['equal'] and r['equal_float32'] and r['max_abs']>0


def test_changed_shape_is_not_fidelity():
    assert not compare_arrays(np.ones((3,2)),np.ones((3,3)))['equal']


@pytest.mark.parametrize('b',[np.array([2.,np.nan]),np.array([1.,0.])])
def test_changed_values_or_missingness_rejected(b):
    assert not compare_arrays(np.array([1.,np.nan]),b)['equal_float32']


def test_matching_missing_values():
    r=compare_arrays(np.array([1.,np.nan]),np.array([1.,np.nan]))
    assert r['equal'] and r['same_nonfinite']


def test_actual_adapter_stops_before_training():
    import os
    import importlib.util
    import pandas as pd
    from examples.advanced.diagnostics.adapter_fidelity_probe import capture
    path=os.environ.get('FEDOT_H9_ADAPTER')
    if not path:
        pytest.skip('Requires frozen host AMLB adapter')
    spec=importlib.util.spec_from_file_location('h9_contract_adapter',path)
    adapter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    x=np.random.RandomState(7).normal(size=(60,4)).astype('float32')
    y=np.arange(60)%3
    for features in (x,pd.DataFrame(x)):
        captured=capture(adapter,features,y)
        assert np.array_equal(captured['data'].idx,np.arange(60))
        assert np.array_equal(captured['data'].target.ravel(),y)
        assert captured['data'].features.shape==(60,4)
