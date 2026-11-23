"""H11: one fixed feature-subsampling proposal, actual API, new CV split.

This is a new-split replication on already explored outer-TRAIN data, not an
untouched dataset or an outer AMLB result. No selection/list extension here.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.adapter_cv_probe import load_adapter, run_pair
from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
from examples.advanced.diagnostics.cache_cost_probe import worker, sha, write_json, probability_loss

LABELS = ('fresh_native', 'fresh_subspace')
SPLIT_SEED = 104736
PROPOSAL = {'colsample_bylevel': .5}


def quality_gate(native, candidate, native_folds, candidate_folds):
    return bool(candidate <= .99 * native and
                all(c <= 1.02 * n for n, c in zip(native_folds, candidate_folds)))


def graph_params(graph):
    prefix, params = graph.split('/n_catboost_', 1)
    assert prefix == '(/n_scaling;)', 'Unexpected graph/edges'
    return ast.literal_eval(params)


def cell(stage, label):
    import pandas as pd
    assert label in LABELS
    with np.load(stage/'train.npz', allow_pickle=False) as z:
        y = z['target'].copy()
    captured = capture(load_adapter(stage), pd.read_pickle(stage/'train-frame.pkl'), y)
    write_json(stage/'cells'/(label+'.input.json'),
               dict(schema=summarize(captured['data']), params=captured['params']))
    worker(stage, label, spec={'cap':600, 'split_seed':SPLIT_SEED,
                             'params':PROPOSAL if label == LABELS[1] else {}},
           input_data=captured['data'])


def audit(stage):
    with np.load(stage/'train.npz', allow_pickle=False) as z:
        y = z['target'].ravel()
    assert len(y) == 7413
    splits = list(StratifiedKFold(3, shuffle=True, random_state=SPLIT_SEED).split(np.zeros(len(y)), y))
    previous = json.loads((stage/'h10-native.json').read_text())
    native_params = graph_params(previous['graph_id_before'])
    results = {}
    for label in LABELS:
        path = stage/'cells'/(label+'.json')
        record = json.loads(path.read_text())
        process = json.loads(path.with_suffix('.process.json').read_text())
        assert process['exit_code'] == 0 and 0 < process['seconds'] <= 665
        assert record['valid'] and len(record['folds']) == 3 and not record['cache_enabled']
        assert record['split_seed'] == SPLIT_SEED
        assert 0 < record['seconds'] <= process['seconds']
        assert record['evaluation_cap'] == 600
        assert record['data_sha256'] == sha(stage/'train.npz')
        assert record['npz_sha256'] == sha(path.with_suffix('.npz'))
        expected = {**native_params, **(PROPOSAL if label == LABELS[1] else {})}
        assert record['graph_id_before'] == record['graph_id_after']
        assert graph_params(record['graph_id_before']) == expected
        assert record['parameter_override'] == (PROPOSAL if label == LABELS[1] else {})
        losses, trees, categories = [], [], []
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            assert np.array_equal(z['classes'], np.unique(y))
            for i, (train, valid) in enumerate(splits):
                assert np.array_equal(z[f'rows_{i}'], valid)
                assert np.array_equal(z[f'train_rows_{i}'], train)
                assert np.array_equal(z[f'truth_{i}'], y[valid])
                losses.append(probability_loss(z[f'probs_{i}'], y[valid], z['classes']))
                fitted = [n for n in record['folds'][i]['fitted_nodes'] if n['operation'] == 'catboost']
                assert len(fitted) == 1 and 0 < fitted[0]['tree_count_'] <= expected['num_trees']
                cats = fitted[0]['cat_feature_indices']
                assert cats == list(range(715, 800)), 'API categorical contract changed'
                trees.append(fitted[0]['tree_count_']); categories.append(len(cats))
        cv = float(np.mean(losses))
        assert abs(cv-record['fitness']) < 1e-10
        results[label] = dict(cv=cv, folds=losses, seconds=record['seconds'],
                              process_seconds=process['seconds'], trees=trees,
                              categorical_counts=categories, prediction_sha256=record['npz_sha256'])
    a,b = [results[label] for label in LABELS]
    return dict(cells=results, split_seed=SPLIT_SEED, relative_improvement=1-b['cv']/a['cv'],
                gate_passed=quality_gate(a['cv'],b['cv'],a['folds'],b['folds']),
                scope='New-split replication on explored outer-TRAIN data; no outer or untouched-data claim')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['suite','worker','audit'])
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--label')
    a = p.parse_args()
    if a.mode == 'suite':
        run_pair(a.stage, LABELS, audit, Path(__file__).resolve())
    elif a.mode == 'worker':
        cell(a.stage, a.label)
    else:
        print(json.dumps(audit(a.stage)))
