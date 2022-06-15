from functools import partial
from numbers import Real

import numpy as np
import pytest

from fedot.core.dag.graph import Graph
from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import default_log
from fedot.core.optimisers.objective import Objective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator, OneFoldInputDataSplit
from test.unit.models.test_model import classification_dataset
from test.unit.validation.test_table_cv import sample_pipeline

_ = classification_dataset


def empty_pipeline():
    return Pipeline()


def throwing_exception_metric(graph: Graph, **kwargs) -> Real:
    x = 1 / 0
    return x


def empty_datasource():
    task = Task(TaskTypesEnum.classification)
    features = np.array([])
    target = np.array([])
    input_data = InputData(idx=[], features=features, target=target,
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData(was_preprocessed=False))
    yield input_data, input_data


def test_pipeline_objective_evaluate_with_different_metrics(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metrics = [ClassificationMetricsEnum.ROCAUC_penalty,
               ClassificationMetricsEnum.accuracy,
               ClassificationMetricsEnum.logloss,
               ClassificationMetricsEnum.ROCAUC,
               ClassificationMetricsEnum.precision,
               ClassificationMetricsEnum.f1]
    for metric in metrics:
        objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
        fitness = objective_eval(pipeline)
        assert fitness.valid


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


def test_pipeline_objective_evaluate_with_timelimit(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    timelimit = 0.0001
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=timelimit, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid

    timelimit = 300
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=timelimit, log=log)
    fitness = objective_eval(pipeline)
    assert fitness.valid


def test_pipeline_objective_evaluate_with_empty_metrics(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metrics = []

    objective_eval = PipelineObjectiveEvaluate(Objective(metrics), data_split, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid


def test_pipeline_objective_evaluate_with_throwing_exception_metrics(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = throwing_exception_metric

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid


def test_pipeline_objective_evaluate_with_empty_datasource(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    data_split = empty_datasource
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
    fitness = objective_eval(pipeline)
    assert not fitness.valid
