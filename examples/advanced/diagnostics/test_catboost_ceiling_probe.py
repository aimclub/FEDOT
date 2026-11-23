import json
import os

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.catboost_ceiling_probe import audit, LABEL, PROPOSAL
from examples.advanced.diagnostics.cache_cost_probe import sha, probability_loss
from examples.advanced.diagnostics.subspace_validation_probe import SPLIT_SEED


def fixture(stage,fault=None):
    (stage/'cells').mkdir()
    y=np.arange(7413)%7
    np.savez(stage/'train.npz',features=np.zeros((len(y),2)),target=y)
    for candidate,path in [(False,stage/'baseline.json'),(True,stage/'cells'/(LABEL+'.json'))]:
        correct=.82 if candidate else .8
        arrays={'classes':np.arange(7)}
        for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y)):
            p=np.full((len(valid),7),(1-correct)/6);p[np.arange(len(valid)),y[valid]]=correct
            if fault=='probs' and candidate:p[0]=0
            arrays.update({f'rows_{i}':valid[::-1] if fault=='rows' and candidate else valid,
                           f'train_rows_{i}':train,f'truth_{i}':y[valid],f'probs_{i}':p})
        np.savez(path.with_suffix('.npz'),**arrays)
        params={'num_trees':6000 if candidate else 3000}
        if fault=='params' and candidate:params['learning_rate']=.2
        trees=4000 if candidate and fault!='not_activated' else 2999
        record=dict(valid=True,folds=[{'fitted_nodes':[dict(operation='catboost',tree_count_=trees,
                   cat_feature_indices=list(range(715,800)))]}]*3,cache_enabled=False,split_seed=SPLIT_SEED,
                   evaluation_cap=600,seconds=2,data_sha256=sha(stage/'train.npz'),
                   npz_sha256=sha(path.with_suffix('.npz')),parameter_override=PROPOSAL if candidate else {},
                   graph_id_before='(/n_scaling;)/n_catboost_'+str(params),
                   graph_id_after='(/n_scaling;)/n_catboost_'+str(params),
                   fitness=float(np.mean([probability_loss(arrays[f'probs_{i}'],arrays[f'truth_{i}'],arrays['classes'])
                                          for i in range(3)])) if fault!='probs' or not candidate else float(-np.log(correct)))
        path.write_text(json.dumps(record));path.with_suffix('.process.json').write_text(json.dumps(dict(exit_code=0,seconds=3)))
        if not candidate:
            (stage/'h11-independent-audit.json').write_text(json.dumps(dict(cells={'fresh_native':dict(
                cv=record['fitness']+.1 if fault=='reference' else record['fitness'],prediction_sha256=record['npz_sha256'])})))


def test_gate_requires_real_horizon_activation(tmp_path):
    fixture(tmp_path,'not_activated')
    assert not audit(tmp_path)['gate_passed']


def test_valid_bounded_gain(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['params','rows','probs','reference'])
def test_reject_corruption(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises((AssertionError,ValueError)):
        audit(tmp_path)


def test_host_real_override_reaches_constructor(tmp_path):
    if not os.environ.get('FEDOT_H9_ADAPTER'):pytest.skip('Host constructor contract')
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostClassificationImplementation
    from fedot.core.operations.operation_parameters import OperationParameters
    node=PipelineNode('catboost');node.parameters={**node.parameters,**PROPOSAL}
    implementation=FedotCatBoostClassificationImplementation(OperationParameters(**node.parameters))
    params=implementation.model.get_params()
    assert params.get('num_trees',params.get('iterations'))==6000
