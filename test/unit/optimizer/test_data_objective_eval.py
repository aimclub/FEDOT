from functools import partial

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import default_log
from fedot.core.optimisers.objective import Objective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator, OneFoldInputDataSplit
from test.unit.models.test_model import classification_dataset
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth, \
    pipeline_fifth
from test.unit.validation.test_table_cv import sample_pipeline, get_classification_data

_ = classification_dataset


def empty_pipeline():
    return Pipeline()


def throwing_exception_metric(*args, **kwargs):
    raise Exception


def actual_fitness(data_split, pipeline, metric):
    for (train_input, test_input) in data_split():
        pipeline.fit(train_input)

        # predicted = pipeline.predict(test_input)
        metric_function = MetricsRepository().metric_by_id(metric, default_callable=metric)
        metric_value = metric_function(pipeline=pipeline, reference_data=test_input)
    return metric_value


def empty_datasource():
    task = Task(TaskTypesEnum.classification)
    features = np.array([])
    target = np.array([])
    input_data = InputData(idx=[], features=features, target=target,
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData(was_preprocessed=False))
    yield input_data, input_data


@pytest.mark.parametrize(
    'pipeline',
    [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth(), pipeline_fifth()]
)
def test_pipeline_objective_evaluate_with_different_metrics(classification_dataset, pipeline):
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    for metric in ClassificationMetricsEnum:
        objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
        fitness = objective_eval(pipeline)
        act_fitness = actual_fitness(data_split, sample_pipeline(), metric)
        print(fitness.value, act_fitness)
        assert fitness.valid
        assert fitness.value is not None
        assert abs(fitness.value - act_fitness) < 0.016


def test_pipeline_objective_evaluate_with_empty_pipeline(classification_dataset):
    pipeline = empty_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
    with pytest.raises(AttributeError):
        objective_eval(pipeline)


def test_pipeline_objective_evaluate_with_cv_fold(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    cv_fold = partial(tabular_cv_generator, classification_dataset, folds=3)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), cv_fold, log=log)
    fitness = objective_eval(pipeline)
    assert fitness.valid


def test_pipeline_objective_evaluate_with_empty_datasource(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = empty_datasource
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid


def test_pipeline_objective_evaluate_with_time_constraint(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    time_constraint = 0.0001
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=time_constraint, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid

    time_constraint = 300
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=time_constraint, log=log)
    fitness = objective_eval(pipeline)
    assert fitness.valid


@pytest.mark.parametrize(
    'metrics',
    [[],
     throwing_exception_metric]
)
def test_pipeline_objective_evaluate_with_invalid_metrics(classification_dataset, metrics):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)

    objective_eval = PipelineObjectiveEvaluate(Objective(metrics), data_split, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid
