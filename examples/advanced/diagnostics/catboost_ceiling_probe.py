"""H12: one 6000-tree ceiling candidate against frozen native H11 evidence.

Same actual API, TRAIN rows, split and all other parameters. No new control fit,
outer data, list extension or speed comparison against historical timing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.adapter_cv_probe import load_adapter, run_pair
from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
from examples.advanced.diagnostics.cache_cost_probe import worker, sha, write_json, probability_loss
from examples.advanced.diagnostics.subspace_validation_probe import graph_params, quality_gate, SPLIT_SEED

LABEL = 'fresh_ceiling6000'
PROPOSAL = {'num_trees': 6000}


def cell(stage):
    import pandas as pd
    with np.load(stage/'train.npz',allow_pickle=False) as z:
        y=z['target'].copy()
    data=capture(load_adapter(stage),pd.read_pickle(stage/'train-frame.pkl'),y)
    write_json(stage/'cells'/(LABEL+'.input.json'),dict(schema=summarize(data['data']),params=data['params']))
    worker(stage,LABEL,spec={'cap':600,'split_seed':SPLIT_SEED,'params':PROPOSAL},input_data=data['data'])


def audit_cell(stage, path, expected_params, expected_override, max_trees):
    with np.load(stage/'train.npz',allow_pickle=False) as z:
        y=z['target'].ravel()
    assert len(y)==7413
    record=json.loads(path.read_text())
    process=json.loads(path.with_suffix('.process.json').read_text())
    assert record['valid'] and len(record['folds'])==3 and not record['cache_enabled']
    assert record['split_seed']==SPLIT_SEED and record['evaluation_cap']==600
    assert process['exit_code']==0 and 0<record['seconds']<=process['seconds']<=665
    assert record['data_sha256']==sha(stage/'train.npz')
    assert record['npz_sha256']==sha(path.with_suffix('.npz'))
    assert record['graph_id_before']==record['graph_id_after']
    assert graph_params(record['graph_id_before'])==expected_params
    assert record['parameter_override']==expected_override
    losses=[];trees=[]
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        assert np.array_equal(z['classes'],np.unique(y))
        for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y)):
            assert np.array_equal(z[f'rows_{i}'],valid)
            assert np.array_equal(z[f'train_rows_{i}'],train)
            assert np.array_equal(z[f'truth_{i}'],y[valid])
            losses.append(probability_loss(z[f'probs_{i}'],y[valid],z['classes']))
            fitted=[n for n in record['folds'][i]['fitted_nodes'] if n['operation']=='catboost']
            assert len(fitted)==1
            assert fitted[0]['cat_feature_indices']==list(range(715,800))
            assert 0<fitted[0]['tree_count_']<=max_trees
            trees.append(fitted[0]['tree_count_'])
    loss=float(np.mean(losses));assert abs(loss-record['fitness'])<1e-10
    return dict(cv=loss,folds=losses,trees=trees,seconds=record['seconds'],
                process_seconds=process['seconds'],prediction_sha256=record['npz_sha256'])


def audit(stage):
    baseline_path=stage/'baseline.json'
    native=json.loads(baseline_path.read_text())
    params=graph_params(native['graph_id_before'])
    assert params['num_trees']==3000 and 'colsample_bylevel' not in params
    baseline=audit_cell(stage,baseline_path,params,{},3000)
    expected=json.loads((stage/'h11-independent-audit.json').read_text())['cells']['fresh_native']
    assert baseline['cv']==expected['cv'] and baseline['prediction_sha256']==expected['prediction_sha256']
    candidate=audit_cell(stage,stage/'cells'/(LABEL+'.json'),{**params,**PROPOSAL},PROPOSAL,6000)
    activated=any(t>3000 for t in candidate['trees'])
    return dict(baseline=baseline,candidate=candidate,split_seed=SPLIT_SEED,
                horizon_activated=activated,relative_improvement=1-candidate['cv']/baseline['cv'],
                gate_passed=activated and quality_gate(baseline['cv'],candidate['cv'],baseline['folds'],candidate['folds']),
                scope='Same-split explored TRAIN data, frozen H11 control; no outer or speed claim')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['suite','worker','audit'])
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--label')
    a=p.parse_args()
    if a.mode=='suite':run_pair(a.stage,(LABEL,),audit,Path(__file__).resolve(),stage_limit=700)
    elif a.mode=='worker':
        assert a.label==LABEL
        cell(a.stage)
    else:print(json.dumps(audit(a.stage)))
