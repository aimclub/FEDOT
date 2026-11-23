from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from examples.advanced.diagnostics.seeded_fit_contract import compare_cells,worker,audit_cell


def test_comparison_rejects_changed_trial_or_lineage():
    first=dict(calls=[dict(graph='a',fitness=.5)],history=[dict(parents=[[0,0]])])
    for field in ['calls','history']:
        broken=deepcopy(first);broken[field]=[]
        with pytest.raises(AssertionError):compare_cells(first,broken)


@pytest.mark.skipif(not os.environ.get('FEDOT_H28_STAGE'),reason='isolated actual model fit host')
def test_real_fit_history_probabilities_and_corruption(tmp_path):
    stage=Path(os.environ['FEDOT_H28_STAGE']);output=tmp_path/'original';output.mkdir()
    worker(stage,output,smoke=True)
    record=audit_cell(output,smoke=True)
    assert record['valid_cv']==6 and record['fits']>=19
    damaged=tmp_path/'damaged';shutil.copytree(output,damaged)
    path=next((damaged/'observation/cv/000').glob('prediction-*.npz'))
    with np.load(path,allow_pickle=False) as z: arrays={key:z[key].copy() for key in z.files}
    arrays['probs'][:]=1/3
    with path.open('wb') as f:np.savez_compressed(f,**arrays)
    with pytest.raises(AssertionError):audit_cell(damaged,smoke=True)
