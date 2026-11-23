import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.advanced.diagnostics.cache_cost_probe import (
    ORDER, audit, cost_gate, probability_loss, sha, write_json,
)


def synthetic_records(path):
    (path / 'cells').mkdir()
    y = np.tile(np.arange(3), 3)
    np.savez(path / 'train.npz', features=np.arange(18).reshape(9, 2), target=y)
    for label in ORDER:
        arrays = {'classes': np.arange(3)}
        for i in range(3):
            rows = np.arange(i * 3, (i + 1) * 3)
            arrays.update({f'rows_{i}': rows, f'train_rows_{i}': np.setdiff1d(np.arange(9), rows),
                           f'truth_{i}': y[rows], f'probs_{i}': np.eye(3) * .7 + .1})
        npz = path / 'cells' / (label + '.npz')
        np.savez(npz, **arrays)
        write_json(npz.with_suffix('.json'), dict(
            valid=True, folds=[{}, {}, {}], data_sha256=sha(path / 'train.npz'),
            npz_sha256=sha(npz), fitness=-np.log(.8), seconds=150 if label.startswith('fresh') else 100,
            graph_id_before='native', graph_id_after='native', node_cache_before={},
            node_cache_after={f'pred_native_default_{i}': 1 for i in range(3)}))


def test_independent_audit_and_gate(tmp_path):
    synthetic_records(tmp_path)
    result = audit(tmp_path)
    assert result['gate_passed']
    assert result['max_abs_probability_difference'] == 0
    assert result['prediction_hits'] == {'cached_a': 3, 'cached_b': 3}


@pytest.mark.parametrize('fault', ['rows', 'classes', 'norm', 'hash', 'graph', 'incomplete'])
def test_bad_suite_is_rejected(tmp_path, fault):
    synthetic_records(tmp_path)
    path = tmp_path / 'cells/fresh_b.json'
    record = json.loads(path.read_text())
    if fault in ('graph', 'incomplete', 'hash'):
        record[{'graph': 'graph_id_after', 'incomplete': 'valid', 'hash': 'npz_sha256'}[fault]] = (
            False if fault == 'incomplete' else 'bad')
    else:
        npz = path.with_suffix('.npz')
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        if fault == 'rows':
            arrays['rows_0'] = arrays['rows_0'][::-1]
        elif fault == 'classes':
            arrays['classes'] = arrays['classes'][::-1]
        else:
            arrays['probs_0'][0, 0] = 0
        np.savez(npz, **arrays)
        record['npz_sha256'] = sha(npz)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit(tmp_path)


@pytest.mark.parametrize('cached,fresh,paired,hits', [
    ([100], [150, 150], True, True), ([100, 101], [120, 150], True, True),
    ([100, 100], [150, float('nan')], True, True),
    ([100, 100], [150, 150], False, True), ([100, 100], [150, 150], True, False),
])
def test_no_admission_for_incomplete_or_unproven_effect(cached, fresh, paired, hits):
    assert not cost_gate(cached, fresh, paired, hits)


def test_runtime_smoke(tmp_path):
    if not os.environ.get('FEDOT_CACHE_COST_SMOKE'):
        pytest.skip('host-native integration is explicitly enabled at admission')
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=90, n_features=6, n_informative=4,
                               n_classes=3, random_state=42)
    np.savez(tmp_path / 'train.npz', features=x, target=y)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cells').mkdir()
    script = Path(__file__).with_name('cache_cost_probe.py')
    for label in ORDER:
        subprocess.run([sys.executable, str(script), 'worker', '--stage', str(tmp_path),
                        '--label', label, '--smoke'], check=True, timeout=120)
    result = audit(tmp_path)
    assert result['max_abs_probability_difference'] <= 1e-8
    assert result['prediction_hits'] == {'cached_a': 3, 'cached_b': 3}
