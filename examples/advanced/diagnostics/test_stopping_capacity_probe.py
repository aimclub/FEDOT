from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import pytest
from examples.advanced.diagnostics.stopping_capacity_probe import candidate_graph,worker,audit_candidate,gate


def test_single_stopping_change():
    g=[dict(operation='lgbm',params=dict(min_data_in_leaf=29,use_eval_set=True,early_stopping_rounds=30),parents=[1]),dict(operation='scaling',params={},parents=[])]
    old=deepcopy(g);new=candidate_graph(g);assert old==g
    assert new[0]['params']['early_stopping_rounds'] is None
    new[0]['params']['early_stopping_rounds']=30;assert new==old


def test_gate_checks_prediction_match_and_every_split():
    c=dict(uniform=[True]*3,active_splits=[False]*3)
    assert gate(c,True) and not gate(c,False)
    c['active_splits'][2]=True;assert not gate(c,True)


@pytest.mark.skipif(not os.environ.get('FEDOT_H32_STAGE'),reason='isolated framework host')
def test_real_eval_no_stopping_and_callback_corruption(tmp_path):
    out=tmp_path/'original';out.mkdir();worker(Path(os.environ['FEDOT_H32_STAGE']),out,smoke=True)
    assert audit_candidate(out,smoke=True)['actual_train_rows']==[32]*3
    bad=tmp_path/'damaged';shutil.copytree(out,bad)
    path=bad/'actual-fit-0.json';r=json.loads(path.read_text());r['callbacks'].append('_EarlyStoppingCallback');path.write_text(json.dumps(r))
    with pytest.raises(AssertionError):audit_candidate(bad,smoke=True)
