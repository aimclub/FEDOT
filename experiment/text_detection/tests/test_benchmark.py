"""Tests for benchmark reports without downloading datasets."""

from types import SimpleNamespace

import pandas as pd
import pytest

from experiment.text_detection import benchmark


class Registry:
    def create(self, dataset_name, split):
        if dataset_name == 'failed':
            raise OSError('Data not available')
        frame = pd.DataFrame({
            'title': [
                'This product has a useful design',
                'A better way to organize your work',
                'An interesting approach to learning',
            ],
            ' num_keywords': [3.0, 4.0, 5.0],
        })
        if dataset_name in ('manual', 'missing'):
            return SimpleNamespace(data=frame, feature_columns=list(frame.columns))
        return SimpleNamespace(
            data=frame,
            feature_columns=list(frame.columns),
            feature_types=['text', 'categorical'],
        )


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = tmp_path / 'labels.csv'
    pd.DataFrame([
        {'dataset': 'manual', 'column': 'title', 'target': 1},
        {'dataset': 'manual', 'column': ' num_keywords', 'target': 0},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(benchmark, 'LABELS_PATH', path)
    monkeypatch.setattr(benchmark, 'REPORTS_DIR', tmp_path / 'reports')

    def set_datasets(*names):
        monkeypatch.setattr(
            benchmark, 'load_benchmark_registry',
            lambda: (Registry(), {name: name for name in names}),
        )

    set_datasets('manual', 'annotated')
    return tmp_path, set_datasets


def test_runner_creates_all_reports(runner):
    tmp_path, _ = runner
    summary = benchmark.evaluate()
    assert set(summary['detector']) == {
        'current_fedot', 'pr903', 'improved_v1', 'improved_v2',
    }
    assert (summary['columns'] == 4).all()
    directory = tmp_path / 'reports'
    for name in (
        'benchmark_predictions.csv', 'benchmark_errors.csv',
        'benchmark_per_dataset.csv', 'benchmark_summary.csv',
        'v2_column_diagnostics.csv', 'missing_labels.csv', 'failed_datasets.csv',
    ):
        assert (directory / name).exists()
    assert pd.read_csv(directory / 'missing_labels.csv').empty
    assert pd.read_csv(directory / 'failed_datasets.csv').empty
    predictions = pd.read_csv(directory / 'benchmark_predictions.csv')
    assert len(predictions) == 16
    assert set(predictions['error_type']) <= {'TP', 'TN', 'FP', 'FN'}


def test_test_split_writes_separate_reports(runner):
    tmp_path, _ = runner
    benchmark.evaluate(split='test')
    assert (tmp_path / 'reports' / 'test' / 'benchmark_summary.csv').exists()
    assert not (tmp_path / 'reports' / 'benchmark_summary.csv').exists()


def test_missing_labels_stop_full_run(runner):
    tmp_path, set_datasets = runner
    set_datasets('manual', 'missing')
    with pytest.raises(RuntimeError, match='Incomplete benchmark'):
        benchmark.evaluate()
    directory = tmp_path / 'reports'
    assert len(pd.read_csv(directory / 'missing_labels.csv')) == 2
    assert not (directory / 'benchmark_summary.csv').exists()


def test_failed_dataset_stops_full_run(runner):
    tmp_path, set_datasets = runner
    set_datasets('manual', 'failed')
    with pytest.raises(RuntimeError, match='Incomplete benchmark'):
        benchmark.evaluate()
    failures = pd.read_csv(tmp_path / 'reports' / 'failed_datasets.csv')
    assert failures['dataset'].tolist() == ['failed']


def test_partial_run_is_explicit(runner):
    _, set_datasets = runner
    set_datasets('manual', 'failed')
    summary = benchmark.evaluate(allow_partial=True)
    assert (summary['columns'] == 2).all()


@pytest.mark.parametrize('kwargs', [
    {'split': 'invalid'},
    {'max_rows': 2},
])
def test_invalid_runner_options(runner, kwargs):
    with pytest.raises(ValueError):
        benchmark.evaluate(**kwargs)


def test_confusion_metrics():
    frame = pd.DataFrame({
        'expected': [1, 1, 0, 0],
        'predicted': [1, 0, 1, 0],
    })
    results = benchmark.metrics(frame)
    assert results['tp'] == 1
    assert results['fp'] == 1
    assert results['fn'] == 1
    assert results['precision'] == 0.5
    assert results['recall'] == 0.5
    assert results['f1'] == 0.5


def test_duplicate_dataset_aliases_processed_once(runner, monkeypatch):
    tmp_path, _ = runner
    monkeypatch.setattr(
        benchmark, 'load_benchmark_registry',
        lambda: (Registry(), {'one': 'manual', 'alias': 'manual'}),
    )
    summary = benchmark.evaluate()
    assert (summary['columns'] == 2).all()
    predictions = pd.read_csv(tmp_path / 'reports' / 'benchmark_predictions.csv')
    assert len(predictions) == 8


def test_failed_rerun_removes_previous_summary(runner):
    tmp_path, set_datasets = runner
    benchmark.evaluate()
    report = tmp_path / 'reports' / 'benchmark_summary.csv'
    assert report.exists()

    set_datasets('manual', 'failed')
    with pytest.raises(RuntimeError, match='Incomplete benchmark'):
        benchmark.evaluate()
    assert not report.exists()


def test_metrics_without_text_columns():
    frame = pd.DataFrame({
        'expected': [0, 0, 0],
        'predicted': [0, 0, 0],
    })
    result = benchmark.metrics(frame)
    assert result['precision'] == 0.0
    assert result['recall'] == 0.0
    assert result['f1'] == 0.0
