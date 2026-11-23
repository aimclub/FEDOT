import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.advanced.diagnostics.fresh_budget_probe import audit, calibrated_cap, trial_fits
from examples.advanced.diagnostics.cache_cost_probe import sha


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_calibration_is_rejected(value):
    with pytest.raises(ValueError):
        calibrated_cap(value)


def test_cost_cap_and_final_reserve():
    assert calibrated_cap(100) == 130
    assert calibrated_cap(200) == 180
    assert not trial_fits(319.9, 130)
    assert trial_fits(320, 130)


def policy_fixture(stage):
    (stage / 'cells').mkdir()
    y = np.tile(np.arange(3), 3)
    np.savez(stage / 'train.npz', features=np.arange(18).reshape(9, 2), target=y)
    params = {'n_jobs': 8, 'num_trees': 3000}
    graph = '(/n_scaling;)/n_catboost_' + repr(params)
    arrays = {'classes': np.arange(3)}
    for i in range(3):
        rows = np.arange(i * 3, (i + 1) * 3)
        arrays.update({f'rows_{i}': rows, f'train_rows_{i}': np.setdiff1d(np.arange(9), rows),
                       f'truth_{i}': y[rows], f'probs_{i}': np.eye(3) * .7 + .1})
    np.savez(stage / 'h7-reference.npz', **arrays)
    (stage / 'h7-independent-audit.json').write_text(json.dumps({
        'cached_seconds': [40, 40], 'independent_losses': {'cached_a': -np.log(.8)},
        'cells': {'cached_a': {'graph_id_before': graph}}}))
    for policy in ('baseline', 'candidate'):
        result = {'status': 'evaluated', 'policy_seconds': 331 if policy == 'candidate' else 181,
                  'valid_proposals': int(policy == 'candidate'), 'selected_params': {},
                  'next_evaluation_cap': 130 if policy == 'candidate' else 52}
        for label in ('initial', 'proposal', 'confirmation'):
            if policy == 'baseline':
                result[label] = {'valid': False, 'process_seconds': 60}
                continue
            name = 'fresh_' + policy + '_' + label
            override = {'colsample_bylevel': .5} if label == 'proposal' else {}
            p = stage / 'cells' / (name + '.json')
            np.savez(p.with_suffix('.npz'), **arrays)
            g = '(/n_scaling;)/n_catboost_' + repr({**params, **override})
            record = dict(valid=True, seconds=100, fitness=-np.log(.8),
                          folds=[{'fitted_nodes': []}] * 3, parameter_override=override,
                          graph_id_before=g, graph_id_after=g, cache_enabled=False,
                          cache_before={}, cache_after={}, data_sha256=sha(stage / 'train.npz'),
                          npz_sha256=sha(p.with_suffix('.npz')))
            p.write_text(json.dumps(record))
            result[label] = dict(name=name, valid=True, evaluation=record, process_seconds=110)
        (stage / (policy + '.json')).write_text(json.dumps(result))


def test_policy_audit_recomputes_metrics_and_feasibility(tmp_path):
    policy_fixture(tmp_path)
    assert audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault', ['budget', 'uncharged', 'cap', 'count'])
def test_policy_accounting_errors_rejected(tmp_path, fault):
    policy_fixture(tmp_path)
    p = tmp_path / 'candidate.json'
    d = json.loads(p.read_text())
    if fault == 'budget':
        d['policy_seconds'] = 601
    elif fault == 'uncharged':
        d['policy_seconds'] = 100
    elif fault == 'cap':
        d['next_evaluation_cap'] = 170
    else:
        d['valid_proposals'] = 2
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        audit(tmp_path)


def test_real_invalid_and_valid_budgeted_cv(tmp_path):
    if not os.environ.get('FEDOT_FRESH_BUDGET_SMOKE'):
        pytest.skip('explicit host-native admission only')
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=90, n_features=6, n_informative=4,
                               n_classes=3, random_state=42)
    np.savez(tmp_path / 'train.npz', features=x, target=y)
    (tmp_path / 'cells').mkdir()
    env = os.environ.copy()
    env.pop('PYTEST_CURRENT_TEST', None)
    script = Path(__file__).with_name('fresh_budget_probe.py')
    for label, cap in [('fresh_invalid', .001), ('fresh_valid', 20)]:
        (tmp_path / 'cells' / (label + '.spec.json')).write_text(json.dumps({'cap': cap, 'params': {}}))
        subprocess.run([sys.executable, str(script), 'worker', '--stage', str(tmp_path),
                        '--label', label, '--smoke'], env=env, check=True, timeout=120)
    invalid = json.loads((tmp_path / 'cells/fresh_invalid.json').read_text())
    valid = json.loads((tmp_path / 'cells/fresh_valid.json').read_text())
    assert not invalid['valid'] and invalid['fitness'] is None
    assert valid['valid'] and len(valid['folds']) == 3 and not valid['cache_enabled']
    assert all(any(n['operation'] == 'catboost' and n['tree_count_'] > 0
                   for n in f['fitted_nodes']) for f in valid['folds'])
