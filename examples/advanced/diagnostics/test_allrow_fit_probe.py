import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.allrow_fit_probe import audit, instrument_fit, LABEL, PROPOSAL
from examples.advanced.diagnostics.cache_cost_probe import worker
from examples.advanced.diagnostics.catboost_ceiling_probe import LABEL as OLD_LABEL
from examples.advanced.diagnostics.subspace_validation_probe import graph_params, SPLIT_SEED
from examples.advanced.diagnostics.test_catboost_ceiling_probe import fixture as ceiling_fixture


def fixture(stage,fault=None):
    ceiling_fixture(stage)
    for suffix in ('.json','.npz','.process.json'):
        (stage/'cells'/(OLD_LABEL+suffix)).rename(stage/'cells'/(LABEL+suffix))
    for candidate,p in [(False,stage/'baseline.json'),(True,stage/'cells'/(LABEL+'.json'))]:
        r=json.loads(p.read_text())
        params=dict(num_trees=3000,use_eval_set=not candidate,learning_rate=.03)
        r['graph_id_before']=r['graph_id_after']='(/n_scaling;)/n_catboost_'+str(params)
        r['parameter_override']=PROPOSAL if candidate else {}
        for fold in r['folds']:fold['fitted_nodes'][0]['tree_count_']=3000 if candidate else 2999
        if candidate and fault=='trees':r['folds'][0]['fitted_nodes'][0]['tree_count_']=2990
        p.write_text(json.dumps(r))
    (stage/'pools').mkdir()
    y=np.arange(7413)%7
    for train,_ in StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y):
        signature=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
        pool=dict(rows=train.tolist(),count=4942,columns=800,categories=list(range(715,800)))
        params=dict(num_trees=3000,use_best_model=False,thread_count=8,learning_rate=.03)
        if fault=='pool_rows':pool['rows']=pool['rows'][::-1]
        if fault=='detector':params['od_wait']=100
        if fault=='categories':pool['categories']=[]
        pools=[pool,pool] if fault=='split' else [pool]
        (stage/'pools'/(signature+'.json')).write_text(json.dumps(dict(pools=pools,model_params=params,trees=3000,callback_count=1)))


def test_audit_complete(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['trees','pool_rows','detector','categories','split'])
def test_audit_rejects_invalid_fit_policy(tmp_path,fault):
    fixture(tmp_path,fault)
    with pytest.raises(AssertionError):audit(tmp_path)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='host-only actual adapter/model')
def test_actual_api_allrow_cv_and_pool_recording(tmp_path):
    import importlib.util
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    from examples.advanced.diagnostics.adapter_fidelity_probe import capture
    spec=importlib.util.spec_from_file_location('h14_test_adapter',os.environ['FEDOT_H9_ADAPTER'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=np.random.RandomState(104740).normal(size=(60,4)).astype('float32')
    x[:,2:]=np.arange(120).reshape(60,2)%3
    y=np.arange(60)%3
    np.savez(tmp_path/'train.npz',features=x,target=y)
    (tmp_path/'cells').mkdir();(tmp_path/'pools').mkdir()
    data=capture(module,x,y)['data']
    original=instrument_fit(tmp_path)
    try:
        worker(tmp_path,'fresh_allrow_smoke',smoke=True,
               spec={'cap':30,'split_seed':SPLIT_SEED,'params':{**PROPOSAL,'enable_categorical':True}},input_data=data)
    finally:FedotCatBoostImplementation.fit=original
    r=json.loads((tmp_path/'cells/fresh_allrow_smoke.json').read_text())
    assert r['valid'] and len(r['folds'])==3
    assert graph_params(r['graph_id_before'])['use_eval_set'] is False
    files=list((tmp_path/'pools').glob('*.json'));assert len(files)==3
    for p in files:
        r=json.loads(p.read_text());assert len(r['pools'])==1 and r['pools'][0]['count']==40
        assert r['pools'][0]['categories']==[2,3] and r['trees']==8
        assert r['callback_count']==1 and r['model_params']['thread_count']==8
