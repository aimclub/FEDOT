from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from examples.advanced.diagnostics.leaf_capacity_probe import candidate_graph,gate,worker,audit_candidate


def test_only_one_declared_parameter_changes():
    original=[dict(operation='lgbm',params=dict(min_data_in_leaf=29,n_jobs=8),parents=[1]),dict(operation='scaling',params={},parents=[])]
    copied=deepcopy(original);changed=candidate_graph(original)
    assert original==copied
    changed[0]['params']['min_data_in_leaf']=29
    assert changed==original


@pytest.mark.parametrize('losses,active', [([1.1,.7,.7],[True]*3),([.9]*3,[False,True,True]),([float('nan')]*3,[True]*3)])
def test_mean_improvement_cannot_hide_failed_gate(losses,active):
    assert not gate([1]*3,losses,active)


@pytest.mark.skipif(not os.environ.get('FEDOT_H30_STAGE'),reason='isolated frozen API model host')
def test_real_cv_and_corruption(tmp_path):
    original=tmp_path/'original';original.mkdir()
    worker(Path(os.environ['FEDOT_H30_STAGE']),original,smoke=True)
    result=audit_candidate(original,smoke=True);assert len(result['fold_losses'])==3
    broken=tmp_path/'damaged';shutil.copytree(original,broken)
    path=broken/'candidate.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['rows_0'][0]=-1
    with path.open('wb') as f:np.savez_compressed(f,**arrays)
    with pytest.raises(AssertionError):audit_candidate(broken,smoke=True)
