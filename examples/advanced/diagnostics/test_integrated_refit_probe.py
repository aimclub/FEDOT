import os
from pathlib import Path
import json
import numpy as np
import pytest

from examples.advanced.diagnostics.integrated_refit_probe import check_final_constructor


def test_allrow_constructor_keeps_parameters_and_drops_aliases():
    expected=dict(n_jobs=8,num_trees=3000,use_eval_set=False,use_best_model=True,learning_rate=.03,od_wait=100)
    actual=dict(thread_count=8,num_trees=3000,use_best_model=False,learning_rate=.03)
    check_final_constructor(actual,expected,True)
    for key,value in [('num_trees',2999),('learning_rate',.1),('thread_count',1),('od_wait',100)]:
        with pytest.raises(AssertionError):check_final_constructor({**actual,key:value},expected,True)


def test_native_constructor_retains_detector_and_best_model():
    expected=dict(n_jobs=8,num_trees=3000,use_eval_set=True,use_best_model=True,od_wait=100)
    actual=dict(thread_count=8,num_trees=3000,use_best_model=True,od_wait=100)
    check_final_constructor(actual,expected,False)
    with pytest.raises(AssertionError):check_final_constructor({**actual,'use_best_model':False},expected,False)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='actual isolated API/tuner/refit host')
def test_actual_integrated_cv_recorder_and_final_refit(tmp_path):
    import pandas as pd
    from examples.advanced.diagnostics.integrated_refit_probe import run_case,audit
    (tmp_path/'harness').symlink_to(Path(os.environ['FEDOT_H9_ADAPTER']).parents[2],target_is_directory=True)
    x=np.random.RandomState(104735).normal(size=(30,4)).astype('float32')
    x[:,2:]=np.arange(60).reshape(30,2)%3;y=np.arange(30)%3
    np.savez(tmp_path/'train.npz',features=x,target=y)
    pd.DataFrame(x,columns=['a','b','c','d']).to_pickle(tmp_path/'train-frame.pkl')
    run_case(tmp_path,smoke=True)
    assert audit(tmp_path,smoke=True)['integration_valid']
    p=tmp_path/'cells/proposal-fold0.json';original=p.read_text();r=json.loads(original)
    r['trees']=11;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError):audit(tmp_path,smoke=True)
    p.write_text(original)
