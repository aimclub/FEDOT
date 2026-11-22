"""Independent read-only AMLB probability audit; write only a NEW audit report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run):
    if not (run / 'COMPLETE').exists() or (run / 'FAILED').exists():
        raise ValueError('Not a valid COMPLETE run: ' + str(run))
    scores = list((run / 'results').rglob('scores/results.csv'))
    predictions = list((run / 'results').rglob('predictions.csv'))
    if len(scores) != 1 or len(predictions) != 1:
        raise ValueError('Expected exactly one score and prediction artifact')
    score = pd.read_csv(scores[0], keep_default_na=False)
    if len(score) != 1 or score.iloc[0].get('info', ''):
        raise ValueError('Invalid score/info')
    row = score.iloc[0]
    for key in ('duration', 'training_duration'):
        if not 0 <= float(row[key]) <= 3600:
            raise ValueError('Full-job budget violation: ' + key)
    frame = pd.read_csv(predictions[0], dtype={'truth': str, 'predictions': str})
    columns = list(frame.columns)
    classes = sorted(c for c in columns if c not in ('truth', 'predictions'))
    truth = frame['truth'].to_numpy()
    if not set(truth).issubset(classes):
        raise ValueError('Truth does not match probability columns')
    probabilities = frame[classes].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid probability entries')
    norm_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
    if norm_error > 1e-6:
        raise ValueError('Invalid probability normalization')
    loss = float(log_loss(truth, probabilities, labels=classes))
    # AMLB writes aggregate scores to six decimal places; predictions retain
    # more precision. Permit only half a unit in that last stored decimal.
    if not np.isclose(loss, float(row['logloss']), atol=5.00001e-7, rtol=0):
        raise ValueError('Recomputed loss differs from AMLB score')
    history = list((run / 'results').rglob('history.json'))
    result = dict(rows=len(truth), classes=classes, csv_columns=columns, logloss=loss,
                  job_seconds=float(row['duration']), training_seconds=float(row['training_duration']),
                  normalization_error=norm_error, prediction_sha256=sha(predictions[0]),
                  truth_sha256=hashlib.sha256(''.join(str(v) + '\n' for v in truth).encode('utf-8')).hexdigest(),
                  score_sha256=sha(scores[0]), histories={str(p.relative_to(run)): sha(p) for p in history})
    return result, truth


def audit_pair(stage):
    results, labels = {}, {}
    for variant in ('baseline', 'candidate'):
        runs = [p for p in (stage / 'runs' / variant).iterdir() if p.is_dir()]
        if len(runs) != 1:
            raise ValueError('Expected exactly one task per variant')
        results[variant], labels[variant] = audit_run(runs[0])
    if not np.array_equal(labels['baseline'], labels['candidate']):
        raise ValueError('Paired truth rows differ')
    if results['baseline']['csv_columns'] != results['candidate']['csv_columns']:
        raise ValueError('Paired prediction column order differs')
    return dict(paired_truth=True, paired_column_order=True, **results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_pair(args.stage)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))
