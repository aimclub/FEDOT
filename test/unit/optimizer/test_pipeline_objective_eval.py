import datetime
from copy import deepcopy

import numpy as np
import pytest
from golem.core.optimisers.fitness import SingleObjFitness

from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.integration.models.test_model import classification_dataset, classification_dataset_with_str_labels
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline
from test.unit.validation.test_table_cv import sample_pipeline
from test.unit.validation.test_time_series_cv import configure_experiment

_ = classification_dataset


def pipeline_first_test():
    pipeline = PipelineBuilder().add_node('rf').add_node('rf').build()
    return pipeline


def pipeline_second_test():
    pipeline = PipelineBuilder().add_node('knn').add_node('knn').build()
    return pipeline


def pipeline_third_test():
    pipeline = PipelineBuilder().add_node('xgboost').build()
    return pipeline


def empty_pipeline():
    return Pipeline()


def throwing_exception_metric(*args, **kwargs):
    raise Exception


def actual_fitness(data_split, pipeline, metric):
    metric_values = []
    for (train_data, test_data) in data_split():
        pipeline.fit(train_data)
        metric_function = MetricsRepository().metric_by_id(metric, default_callable=metric)
        metric_values.append(metric_function(pipeline=pipeline, reference_data=test_data))
    mean_metric = np.mean(metric_values, axis=0)
    return SingleObjFitness(mean_metric)


def empty_datasource():
    task = Task(TaskTypesEnum.classification)
    features = np.array([])
    target = np.array([])
    input_data = InputData(idx=[], features=features, target=target,
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData())
    yield input_data, input_data


@pytest.mark.parametrize(
    'pipeline',
    [pipeline_first_test(), pipeline_second_test(), pipeline_third_test()]
)
def test_pipeline_objective_evaluate_with_different_metrics(classification_dataset, pipeline):
    for metric in ClassificationMetricsEnum:
        data_producer = DataSourceSplitter(cv_folds=None).build(classification_dataset)
        check_pipeline = deepcopy(pipeline)
        objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric),
                                                   data_producer=data_producer)
        fitness = objective_eval(pipeline)
        act_fitness = actual_fitness(data_producer, check_pipeline, metric)
        assert fitness.valid
        assert fitness.value is not None
        assert np.isclose(fitness.value, act_fitness.value, atol=1e-8), metric.name


@pytest.mark.parametrize(
    'pipeline',
    [pipeline_first_test(), pipeline_second_test(), pipeline_third_test()]
)
def test_pipeline_objective_evaluate_with_different_metrics_with_str_labes(pipeline):
    for metric in ClassificationMetricsEnum:
        one_fold_split = OneFoldInputDataSplit()
        data_split = partial(one_fold_split.input_split, input_data=classification_dataset_with_str_labels())
        check_pipeline = deepcopy(pipeline)
        objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric), data_split)
        fitness = objective_eval(pipeline)
        act_fitness = actual_fitness(data_split, check_pipeline, metric)
        assert fitness.valid
        assert fitness.value is not None
        assert np.isclose(fitness.value, act_fitness.value, atol=1e-8), metric.name


def test_pipeline_objective_evaluate_with_empty_pipeline(classification_dataset):
    pipeline = empty_pipeline()
    data_producer = DataSourceSplitter(cv_folds=None).build(classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric),
                                               data_producer=data_producer)
    with pytest.raises(AttributeError):
        objective_eval(pipeline)


def test_pipeline_objective_evaluate_with_cv_fold(classification_dataset):
    pipeline = sample_pipeline()

    data_producer = DataSourceSplitter(cv_folds=5).build(classification_dataset)
    metric = ClassificationMetricsEnum.logloss

    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric),
                                               data_producer=data_producer)
    fitness = objective_eval(pipeline)
    assert fitness.valid
    assert fitness.value is not None


def test_pipeline_objective_evaluate_with_empty_datasource(classification_dataset):
    with pytest.raises(ValueError):
        pipeline = sample_pipeline()

        data_split = empty_datasource
        metric = ClassificationMetricsEnum.ROCAUC_penalty

        objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric), data_split)
        objective_eval(pipeline)


def test_pipeline_objective_evaluate_with_time_constraint(classification_dataset):
    pipeline = sample_pipeline()

    data_producer = DataSourceSplitter(cv_folds=None).build(classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    time_constraint = datetime.timedelta(seconds=0.0001)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric),
                                               data_producer=data_producer,
                                               time_constraint=time_constraint)
    fitness = objective_eval(pipeline)
    assert not fitness.valid

    time_constraint = datetime.timedelta(seconds=300)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metric),
                                               data_producer=data_producer,
                                               time_constraint=time_constraint)
    fitness = objective_eval(pipeline)
    assert fitness.valid
    assert fitness.value is not None


@pytest.mark.parametrize(
    'metrics',
    [[],
     throwing_exception_metric]
)
def test_pipeline_objective_evaluate_with_invalid_metrics(classification_dataset, metrics):
    with pytest.raises(Exception):
        pipeline = sample_pipeline()

        data_producer = DataSourceSplitter(cv_folds=None).build(classification_dataset)
        objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metrics),
                                                   data_producer=data_producer)
        objective_eval(pipeline)


@pytest.mark.parametrize('folds, actual_value', [(2, 9.8965), (3, 38.624)])
def test_pipeline_objective_evaluate_for_timeseries_cv(folds, actual_value):
    forecast_len, validation_blocks, time_series = configure_experiment()
    objective = MetricsObjective(RegressionMetricsEnum.MSE)
    data_producer = DataSourceSplitter(folds, validation_blocks).build(time_series)
    simple_pipeline = get_simple_ts_pipeline()
    objective_evaluate = PipelineObjectiveEvaluate(objective, data_producer, validation_blocks=validation_blocks)
    metric_value = objective_evaluate.evaluate(simple_pipeline).value
    assert np.isclose(metric_value, actual_value)
