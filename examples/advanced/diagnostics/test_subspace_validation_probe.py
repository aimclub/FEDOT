import json
import os

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.subspace_validation_probe import (
    audit, quality_gate, SPLIT_SEED, LABELS, PROPOSAL)
from examples.advanced.diagnostics.cache_cost_probe import sha, worker


def test_gain_and_fold_guard_are_both_required():
    assert quality_gate(1., .98, [1,1,1], [.98,.98,.98])
    assert not quality_gate(1., .995, [1,1,1], [.995,.995,.995])
    assert not quality_gate(1., .98, [1,1,1], [1.03,.955,.955])


def fixture(stage, fault=None):
    (stage/'cells').mkdir()
    y = np.arange(7413)%7
    np.savez(stage/'train.npz',features=np.zeros((len(y),2)),target=y)
    graph = "(/n_scaling;)/n_catboost_{'num_trees': 3000}"
    (stage/'h10-native.json').write_text(json.dumps(dict(graph_id_before=graph)))
    for label in LABELS:
        params = {'num_trees':3000, **(PROPOSAL if label==LABELS[1] else {})}
        if fault == 'params': params['max_depth']=9
        arrays = {'classes':np.arange(7)}
        for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y)):
            p = np.full((len(valid),7),.2/6);p[np.arange(len(valid)),y[valid]]=.8
            if fault=='probabilities':p[0]=0
            arrays.update({f'rows_{i}':valid[::-1] if fault=='rows' else valid,
                           f'train_rows_{i}':train,f'truth_{i}':y[valid],f'probs_{i}':p})
        p = stage/'cells'/(label+'.npz');np.savez(p,**arrays)
        rec = dict(valid=True,folds=[{'fitted_nodes':[dict(operation='catboost',tree_count_=100,
                     cat_feature_indices=[] if fault=='categories' else list(range(715,800)))]}]*3,
                   cache_enabled=False,split_seed=42 if fault=='seed' else SPLIT_SEED,
                   seconds=1,evaluation_cap=600,data_sha256=sha(stage/'train.npz'),npz_sha256=sha(p),
                   graph_id_before='(/n_scaling;)/n_catboost_'+str(params),
                   graph_id_after='(/n_scaling;)/n_catboost_'+str(params),
                   parameter_override=PROPOSAL if label==LABELS[1] else {}, fitness=float(-np.log(.8)))
        p.with_suffix('.json').write_text(json.dumps(rec))
        p.with_suffix('.process.json').write_text(json.dumps(dict(exit_code=0,seconds=2)))


def test_paired_audit(tmp_path):
    fixture(tmp_path)
    assert not audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['params','rows','probabilities','categories','seed'])
def test_corrupted_evidence_rejected(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises((AssertionError,ValueError)):
        audit(tmp_path)


def test_host_new_split_is_used_by_real_evaluator(tmp_path):
    if not os.environ.get('FEDOT_H9_ADAPTER'):pytest.skip('Host native-CV contract')
    x = np.random.RandomState(12).normal(size=(60,4)).astype('float32');y=np.arange(60)%3
    np.savez(tmp_path/'train.npz',features=x,target=y);(tmp_path/'cells').mkdir()
    worker(tmp_path,'fresh_seed_smoke',smoke=True,spec={'cap':30,'split_seed':SPLIT_SEED,'params':PROPOSAL})
    record=json.loads((tmp_path/'cells/fresh_seed_smoke.json').read_text())
    assert record['valid'] and record['split_seed']==SPLIT_SEED
    with np.load(tmp_path/'cells/fresh_seed_smoke.npz') as z:
        for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y)):
            assert np.array_equal(z[f'rows_{i}'],valid)
            assert np.array_equal(z[f'train_rows_{i}'],train)
