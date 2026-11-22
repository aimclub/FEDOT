import numpy as np
import pandas as pd
import pytest

from examples.advanced.diagnostics.audit_followup import audit_pair


def make_pair(tmp_path):
    for variant in ('baseline', 'candidate'):
        run = tmp_path / 'runs' / variant / 'toy-fold6-small'
        scores = run / 'results/session/scores'
        predictions = run / 'results/session/predictions/toy/6'
        scores.mkdir(parents=True)
        predictions.mkdir(parents=True)
        (run / 'COMPLETE').write_text('test-marker')
        pd.DataFrame({'2': [.05, .05, .9], '1': [.9, .05, .05], '10': [.05, .9, .05],
                      'truth': ['1', '10', '2'], 'predictions': ['1', '10', '2']}).to_csv(
            predictions / 'predictions.csv', index=False)
        pd.DataFrame([dict(logloss=-np.log(.9), duration=100, training_duration=90, info='')]).to_csv(
            scores / 'results.csv', index=False)
    return tmp_path


def test_lexical_class_alignment_and_paired_truth(tmp_path):
    result = audit_pair(make_pair(tmp_path))
    assert result['paired_truth'] and result['paired_column_order']
    assert result['baseline']['classes'] == ['1', '10', '2']
    assert result['candidate']['logloss'] == pytest.approx(-np.log(.9))
    assert result['candidate']['prediction_sha256'] == result['baseline']['prediction_sha256']


def test_six_decimal_amlb_score_rounding(tmp_path):
    stage = make_pair(tmp_path)
    for score in stage.rglob('scores/results.csv'):
        frame = pd.read_csv(score, keep_default_na=False)
        frame['logloss'] = frame['logloss'].round(6)
        frame.to_csv(score, index=False)
    result = audit_pair(stage)
    assert result['baseline']['logloss'] == pytest.approx(-np.log(.9))


@pytest.mark.parametrize('fault', ['budget', 'info', 'truth', 'columns', 'norm', 'failed', 'score'])
def test_bad_results_are_rejected(tmp_path, fault):
    stage = make_pair(tmp_path)
    run = stage / 'runs/candidate/toy-fold6-small'
    score = run / 'results/session/scores/results.csv'
    prediction = run / 'results/session/predictions/toy/6/predictions.csv'
    if fault == 'score':
        frame = pd.read_csv(score, keep_default_na=False)
        frame.loc[0, 'logloss'] += 1e-6
        frame.to_csv(score, index=False)
    elif fault in ('budget', 'info'):
        frame = pd.read_csv(score, keep_default_na=False)
        frame.loc[0, 'duration' if fault == 'budget' else 'info'] = 3601 if fault == 'budget' else 'fit failed'
        frame.to_csv(score, index=False)
    elif fault == 'failed':
        (run / 'FAILED').write_text('exit=1')
    else:
        frame = pd.read_csv(prediction)
        if fault == 'truth':
            frame.loc[0, 'truth'] = 2
        elif fault == 'columns':
            frame = frame[frame.columns[::-1]]
        else:
            frame.loc[0, '1'] = .5
        frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError):
        audit_pair(stage)
