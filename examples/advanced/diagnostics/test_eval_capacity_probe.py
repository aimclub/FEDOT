from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import pytest
from examples.advanced.diagnostics.eval_capacity_probe import candidate_graph,worker,audit_candidate


def test_single_switch_retains_leaf_size_and_esr_in_graph():
    g=[dict(operation='lgbm',params=dict(min_data_in_leaf=29,use_eval_set=True,early_stopping_rounds=30),parents=[1]),dict(operation='scaling',params={},parents=[])]
    old=deepcopy(g);new=candidate_graph(g)
    assert g==old and new[0]['params']['min_data_in_leaf']==29
    new[0]['params']['use_eval_set']=True;assert new==old


@pytest.mark.skipif(not os.environ.get('FEDOT_H31_STAGE'),reason='isolated framework host')
def test_real_noeval_cv_and_constructor_corruption(tmp_path):
    out=tmp_path/'original';out.mkdir();worker(Path(os.environ['FEDOT_H31_STAGE']),out,smoke=True)
    report=audit_candidate(out,smoke=True);assert report['actual_train_rows']==[40]*3
    bad=tmp_path/'damaged';shutil.copytree(out,bad)
    path=bad/'actual-fit-0.json';r=json.loads(path.read_text());r['rows']=32;path.write_text(json.dumps(r))
    with pytest.raises(AssertionError):audit_candidate(bad,smoke=True)
