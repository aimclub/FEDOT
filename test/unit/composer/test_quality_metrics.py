import sys
import pytest

from fedot.core.composer.metrics import QualityMetric
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum,\
    ComplexityMetricsEnum, MetricsRepository, RegressionMetricsEnum
from test.pipeline_manager import default_valid_pipeline
from test.data_manager import multi_target_data_setup, data_setup


def test_structural_quality_correct():
    pipeline = default_valid_pipeline()
    metric_function = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)
    expected_metric_value = 13
    actual_metric_value = metric_function(pipeline)
    assert actual_metric_value <= expected_metric_value


@pytest.mark.parametrize('data_fixture', ['data_setup'])
def test_classification_quality_metric(data_setup):
    train, _ = data_setup
    pipeline = default_valid_pipeline()
    pipeline.fit(input_data=train)

    for metric in ClassificationMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(pipeline=pipeline, reference_data=train)
        assert 0 < abs(metric_value) < sys.maxsize


def test_regression_quality_metric(data_setup):
    train, _ = data_setup
    pipeline = default_valid_pipeline()
    pipeline.fit(input_data=train)

    for metric in RegressionMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(pipeline=pipeline, reference_data=train)
        assert metric_value > 0


def test_data_preparation_for_multi_target_correct(multi_target_data_setup):
    train, test = multi_target_data_setup
    simple_pipeline = Pipeline(PrimaryNode('linear'))
    simple_pipeline.fit(input_data=train)

    source_shape = test.target.shape
    # Get converted data
    results, new_test = QualityMetric()._simple_prediction(simple_pipeline, test)
    number_elements = len(new_test.target)
    assert source_shape[0] * source_shape[1] == number_elements
