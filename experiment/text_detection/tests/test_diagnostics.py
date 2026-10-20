"""Tests for inspecting saved benchmark mistakes."""

import pandas as pd

from experiment.text_detection.diagnostics import display_errors


def test_v2_errors_include_detection_reason(tmp_path, capsys):
    predictions = pd.DataFrame([
        {
            'dataset': 'sample', 'column': 'category',
            'detector': 'improved_v2', 'expected': 0, 'predicted': 1,
        },
        {
            'dataset': 'sample', 'column': 'title',
            'detector': 'improved_v2', 'expected': 1, 'predicted': 1,
        },
    ])
    diagnostics = pd.DataFrame([
        {
            'dataset': 'sample', 'column': 'category',
            'reason': 'DIVERSE_SHORT_TEXT', 'feature_mean_word_count': 3.0,
        },
        {'dataset': 'sample', 'column': 'title', 'reason': 'LONG_TEXT'},
    ])
    predictions.to_csv(tmp_path / 'benchmark_predictions.csv', index=False)
    diagnostics.to_csv(tmp_path / 'v2_column_diagnostics.csv', index=False)

    display_errors(tmp_path, 'improved_v2')

    report = pd.read_csv(tmp_path / 'improved_v2_errors.csv')
    assert report['column'].tolist() == ['category']
    assert report['reason'].tolist() == ['DIVERSE_SHORT_TEXT']
    assert 'category' in capsys.readouterr().out


def test_no_errors_produces_no_extra_file(tmp_path, capsys):
    pd.DataFrame([
        {
            'dataset': 'sample', 'column': 'title',
            'detector': 'improved_v2', 'expected': 1, 'predicted': 1,
        },
    ]).to_csv(tmp_path / 'benchmark_predictions.csv', index=False)
    pd.DataFrame(columns=['dataset', 'column']).to_csv(
        tmp_path / 'v2_column_diagnostics.csv', index=False
    )

    display_errors(tmp_path, 'improved_v2')

    assert 'No errors' in capsys.readouterr().out
    assert not (tmp_path / 'improved_v2_errors.csv').exists()
