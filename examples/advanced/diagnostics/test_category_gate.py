import json

import numpy as np
import pytest

from examples.advanced.diagnostics.category_gate import audit_category_gate, sha, MODEL, HELPER


def prepared(tmp_path, fault=None):
    probe, full, training = [tmp_path / name for name in ('probe', 'full', 'training')]
    training.mkdir()
    (probe / 'screen').mkdir(parents=True)
    for variant in ('baseline', 'candidate'):
        for relative in (MODEL, HELPER):
            for stage in (probe, full):
                path = stage / variant / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(variant + ':' + relative)
    y = np.arange(30) % 3
    np.savez(training / 'train.npz', target=y, features=np.zeros((30, 2)), task_id=168910, fold=6)
    rows = np.arange(24, 30)
    truth = y[rows].copy()
    probabilities = np.eye(3)[truth] * .8 + .2 / 3
    if fault == 'labels':
        truth[0] = 2
    if fault == 'norm':
        probabilities[0] = [0., 0., 0.]
    np.savez(probe / 'screen/candidate.npz', confirm_0=probabilities,
             confirm_0_rows=rows, confirm_0_truth=truth, classes=np.arange(3))
    np.savez(probe / 'screen/splits.npz', confirm=rows[::-1] if fault == 'split' else rows)
    (probe / 'QUEUE_FINISHED').touch()
    (probe / 'launch.json').write_text(json.dumps({'after': str(training)}))
    (probe / 'screen/baseline.log').write_text('Found a category not in the training set')
    report = dict(status='negative' if fault == 'negative' else 'passed',
                  reproduced_on_fold6_training=True, data_sha256=sha(training / 'train.npz'),
                  split_sha256=sha(probe / 'screen/splits.npz'),
                  cells=dict(baseline=dict(status='failed'), candidate=dict(status='complete',
                    evaluation=dict(prediction_sha256=sha(probe / 'screen/candidate.npz')))))
    (probe / 'screen/summary.json').write_text(json.dumps(report))
    if fault == 'source':
        (full / 'candidate' / MODEL).write_text('different candidate')
    return probe, full


def test_valid_gate_requires_independent_data_and_exact_tested_source(tmp_path):
    result = audit_category_gate(*prepared(tmp_path))
    assert result['admitted'] and result['rows'] == 6
    assert result['train_only_confirmation_loss'] == pytest.approx(-np.log(.8 + .2 / 3))


@pytest.mark.parametrize('fault', ['negative', 'source', 'labels', 'norm', 'split'])
def test_failed_gate_or_changed_artifacts_cannot_launch_full_run(tmp_path, fault):
    with pytest.raises(ValueError):
        audit_category_gate(*prepared(tmp_path, fault))
