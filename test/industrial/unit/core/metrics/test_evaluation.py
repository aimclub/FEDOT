import numpy as np
import pytest

from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer


@pytest.fixture()
def basic_metric_data_clf():
    target_label = np.array([1, 2, 3, 4])
    test_label = np.array([1, 1, 2, 4])
    test_proba = np.array([[0.8, 0.1, 0.05, 0.05],
                           [0.8, 0.1, 0.05, 0.05],
                           [0.1, 0.8, 0.05, 0.05],
                           [0.05, 0.1, 0.05, 0.8]])

    return test_proba, test_label, target_label


@pytest.fixture()
def basic_metric_data_reg():
    test_label = np.array([1.1, 2.2, 3.3, 4.4])
    target_label = np.array([10, 0.1, 2.3, 4.1])

    return test_label, target_label


def test_performance_analyzer_tsc(basic_metric_data_clf):
    test_proba, test_label, target_label = basic_metric_data_clf
    performance_analyzer = PerformanceAnalyzer()
    target_metrics = ['roc_auc', 'f1', 'accuracy', 'logloss', 'precision']

    score = performance_analyzer.calculate_metrics(
        target=target_label,
        predicted_labels=test_label,
        predicted_probs=test_proba,
        target_metrics=target_metrics)
    assert score is not None
    assert isinstance(score, dict)
    assert len(score) == len(target_metrics)


def test_performance_analyzer_reg(basic_metric_data_reg):
    test_label, target_label = basic_metric_data_reg
    performance_analyzer = PerformanceAnalyzer()
    target_metrics = ['rmse', 'r2', 'mae', 'mse', 'mape']

    score = performance_analyzer.calculate_metrics(
        target=target_label,
        predicted_labels=test_label,
        target_metrics=target_metrics)
    assert score is not None
    assert isinstance(score, dict)
    assert len(score) == len(target_metrics)
