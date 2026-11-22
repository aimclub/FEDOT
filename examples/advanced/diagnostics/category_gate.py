"""Independent admission audit for the isolated categorical full-run pair."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

MODEL = 'fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py'
HELPER = 'examples/advanced/diagnostics/category_schema.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_category_gate(probe, full_stage):
    if not (probe / 'QUEUE_FINISHED').exists() or (probe / 'QUEUE_FAILED').exists():
        raise ValueError('Category screen is not successfully finished')
    report = json.loads((probe / 'screen/summary.json').read_text())
    if report['status'] != 'passed' or not report['reproduced_on_fold6_training']:
        raise ValueError('Category gate did not pass')
    if report['cells']['baseline']['status'] != 'failed' or report['cells']['candidate']['status'] != 'complete':
        raise ValueError('Expected baseline failure and candidate success')
    if 'Found a category not in the training set' not in (probe / 'screen/baseline.log').read_text(errors='replace'):
        raise ValueError('Missing baseline category error')
    data = Path(json.loads((probe / 'launch.json').read_text())['after']) / 'train.npz'
    if not data.resolve().is_relative_to(probe.parent.resolve()) or sha(data) != report['data_sha256']:
        raise ValueError('Training data identity changed')
    predictions = probe / 'screen/candidate.npz'
    if sha(predictions) != report['cells']['candidate']['evaluation']['prediction_sha256']:
        raise ValueError('Prediction identity changed')
    splits = probe / 'screen/splits.npz'
    if sha(splits) != report['split_sha256']:
        raise ValueError('Split identity changed')
    with np.load(data, allow_pickle=False) as source, np.load(predictions, allow_pickle=False) as result:
        if int(source['task_id']) != 168910 or int(source['fold']) != 6:
            raise ValueError('Wrong task/fold')
        y = source['target'].ravel()
        rows = result['confirm_0_rows']
        with np.load(splits, allow_pickle=False) as partition:
            if not np.array_equal(rows, partition['confirm']):
                raise ValueError('Confirmation row order differs from the frozen split')
        truth, classes, probabilities = result['confirm_0_truth'], result['classes'], result['confirm_0']
        if not np.array_equal(truth, y[rows]) or not np.array_equal(classes, np.unique(y)):
            raise ValueError('Truth/classes mismatch')
        if probabilities.shape != (len(truth), len(classes)) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid probabilities')
        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError('Invalid probability range')
        error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
        if error > 1e-6:
            raise ValueError('Invalid probability normalization')
        loss = float(log_loss(truth, probabilities, labels=classes))
    hashes = {}
    for variant in ('baseline', 'candidate'):
        for relative in (MODEL, HELPER):
            a, b = probe / variant / relative, full_stage / variant / relative
            if sha(a) != sha(b):
                raise ValueError('Full runtime differs from the tested category implementation')
            hashes[variant + '/' + relative] = sha(b)
    return dict(admitted=True, rows=len(truth), classes=classes.tolist(),
                train_only_confirmation_loss=loss, normalization_error=error,
                prediction_sha256=sha(predictions), source_hashes=hashes)
