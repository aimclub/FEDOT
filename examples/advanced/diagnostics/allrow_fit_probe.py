"""H14: one actual-API all-row fixed-horizon fit vs frozen H11 native CV.

Joint sample-use/stopping policy, not a pure sample-size causal intervention.
"""
import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.adapter_cv_probe import load_adapter, run_pair
from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
from examples.advanced.diagnostics.cache_cost_probe import worker, sha, write_json
from examples.advanced.diagnostics.catboost_ceiling_probe import audit_cell
from examples.advanced.diagnostics.subspace_validation_probe import graph_params, quality_gate, SPLIT_SEED

LABEL = 'fresh_allrow3000'
PROPOSAL = {'use_eval_set': False}


def instrument_fit(stage):
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    original_fit = FedotCatBoostImplementation.fit

    @wraps(original_fit)
    def observed_fit(self, input_data):
        pools = []
        original_pool = self.convert_to_pool
        def observed_pool(data, identify_cats):
            pool = original_pool(data, identify_cats)
            pools.append(dict(rows=np.asarray(data.idx).tolist(), count=pool.num_row(),
                              columns=pool.num_col(), categories=pool.get_cat_feature_indices()))
            return pool
        self.convert_to_pool = observed_pool
        try:
            model = original_fit(self, input_data)
            signature = hashlib.sha256(np.asarray(input_data.idx,dtype=np.int64).tobytes()).hexdigest()
            write_json(stage/'pools'/(signature+'.json'),dict(pools=pools, model_params=model.get_params(),
                       trees=int(model.tree_count_),callback_count=len(self.fit_callbacks)))
            return model
        finally:
            del self.convert_to_pool
    FedotCatBoostImplementation.fit = observed_fit
    return original_fit


def cell(stage):
    import pandas as pd
    with np.load(stage/'train.npz',allow_pickle=False) as z:y=z['target'].copy()
    data=capture(load_adapter(stage),pd.read_pickle(stage/'train-frame.pkl'),y)
    write_json(stage/'cells'/(LABEL+'.input.json'),dict(schema=summarize(data['data']),params=data['params']))
    instrument_fit(stage)
    worker(stage,LABEL,spec={'cap':600,'split_seed':SPLIT_SEED,'params':PROPOSAL},input_data=data['data'])


def audit(stage):
    baseline_path=stage/'baseline.json'
    params=graph_params(json.loads(baseline_path.read_text())['graph_id_before'])
    assert params['num_trees']==3000 and params['use_eval_set'] is True and 'colsample_bylevel' not in params
    baseline=audit_cell(stage,baseline_path,params,{},3000)
    reference=json.loads((stage/'h11-independent-audit.json').read_text())['cells']['fresh_native']
    assert baseline['cv']==reference['cv'] and baseline['prediction_sha256']==reference['prediction_sha256']
    candidate=audit_cell(stage,stage/'cells'/(LABEL+'.json'),{**params,**PROPOSAL},PROPOSAL,3000)
    assert candidate['trees']==[3000]*3, 'All-row policy requires complete fixed horizon'
    with np.load(stage/'train.npz',allow_pickle=False) as z:y=z['target'].ravel()
    pool_files=list((stage/'pools').glob('*.json'));assert len(pool_files)==3
    for train,_ in StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y):
        signature=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
        r=json.loads((stage/'pools'/(signature+'.json')).read_text())
        assert len(r['pools'])==1
        pool=r['pools'][0]
        assert pool['rows']==train.tolist() and pool['count']==len(train)==4942
        assert pool['columns']==800 and pool['categories']==list(range(715,800))
        assert r['callback_count']==1 and r['trees']==3000
        actual=r['model_params']
        excluded={'n_jobs','use_eval_set','enable_categorical','callbacks','fit_time_limit',
                  'early_stopping_rounds','od_wait','od_type','od_pval'}
        expected={k:v for k,v in params.items() if k not in excluded}
        expected.update(use_best_model=False,thread_count=8)
        assert all(actual.get(k)==v for k,v in expected.items()), 'Actual constructor differs from frozen graph'
        assert actual['use_best_model'] is False and actual['thread_count']==8
        assert actual['num_trees']==3000 and actual['learning_rate']==params['learning_rate']
        assert not set(actual)&{'early_stopping_rounds','od_wait','od_type','od_pval'}
    return dict(baseline=baseline,candidate=candidate,split_seed=SPLIT_SEED,
                pool_rows_per_fold=[4942]*3,relative_improvement=1-candidate['cv']/baseline['cv'],
                gate_passed=quality_gate(baseline['cv'],candidate['cv'],baseline['folds'],candidate['folds']),
                scope='Joint all-row/fixed-horizon policy; explored TRAIN, frozen control; no outer or speed claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','worker','audit'])
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--label')
    a=p.parse_args()
    if a.mode=='suite':run_pair(a.stage,(LABEL,),audit,Path(__file__).resolve(),stage_limit=700)
    elif a.mode=='worker':
        assert a.label==LABEL
        cell(a.stage)
    else:print(json.dumps(audit(a.stage)))
