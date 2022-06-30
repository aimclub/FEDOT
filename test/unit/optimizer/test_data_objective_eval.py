import pytest
from copy import deepcopy
from functools import partial

import numpy as np


from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import default_log
from fedot.core.optimisers.fitness import SingleObjFitness
from fedot.core.optimisers.objective import Objective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator, OneFoldInputDataSplit
from test.unit.models.test_model import classification_dataset
from test.unit.validation.test_table_cv import sample_pipeline

_ = classification_dataset


def pipeline_first_test():
    pipeline = PipelineBuilder().add_node('rf').add_node('rf').to_pipeline()
    return pipeline


def pipeline_second_test():
    pipeline = PipelineBuilder().add_node('knn').add_node('knn').to_pipeline()
    return pipeline


def pipeline_third_test():
    pipeline = PipelineBuilder().add_node('xgboost').to_pipeline()
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
                           supplementary_data=SupplementaryData(was_preprocessed=False))
    yield input_data, input_data


@pytest.mark.parametrize(
    'pipeline',
    [pipeline_first_test(), pipeline_second_test(), pipeline_third_test()]
)
def test_pipeline_objective_evaluate_with_different_metrics(classification_dataset, pipeline):
    log = default_log(__name__)
    for metric in ClassificationMetricsEnum:
        one_fold_split = OneFoldInputDataSplit()
        data_split = partial(one_fold_split.input_split, input_data=classification_dataset)
        check_pipeline = deepcopy(pipeline)
        objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, log=log)
        fitness = objective_eval(pipeline)
        act_fitness = actual_fitness(data_split, check_pipeline, metric)
        assert fitness.valid
        assert fitness.value is not None
        assert np.isclose(fitness.value, act_fitness.value, atol=1e-8), metric.name


def test_pipeline_objective_evaluate_with_empty_pipeline(classification_dataset):
    pipeline = empty_pipeline()

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split)
    with pytest.raises(AttributeError):
        objective_eval(pipeline)


def test_pipeline_objective_evaluate_with_cv_fold(classification_dataset):
    pipeline = sample_pipeline()

    cv_fold = partial(tabular_cv_generator, classification_dataset, folds=3)
    metric = ClassificationMetricsEnum.logloss

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), cv_fold)
    fitness = objective_eval(pipeline)
    assert fitness.valid
    assert fitness.value is not None


def test_pipeline_objective_evaluate_with_empty_datasource(classification_dataset):
    pipeline = sample_pipeline()

    data_split = empty_datasource
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split)
    fitness = objective_eval(pipeline)
    assert not fitness.valid


def test_pipeline_objective_evaluate_with_time_constraint(classification_dataset):
    pipeline = sample_pipeline()

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)
    metric = ClassificationMetricsEnum.ROCAUC_penalty

    time_constraint = 0.0001
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=time_constraint)
    fitness = objective_eval(pipeline)
    assert not fitness.valid

    time_constraint = 300
    objective_eval = PipelineObjectiveEvaluate(Objective(metric), data_split, time_constraint=time_constraint)
    fitness = objective_eval(pipeline)
    assert fitness.valid
    assert fitness.value is not None


@pytest.mark.parametrize(
    'metrics',
    [[],
     throwing_exception_metric]
)
def test_pipeline_objective_evaluate_with_invalid_metrics(classification_dataset, metrics):
    pipeline = sample_pipeline()

    data_split = partial(OneFoldInputDataSplit().input_split, input_data=classification_dataset)

    objective_eval = PipelineObjectiveEvaluate(Objective(metrics), data_split)
    fitness = objective_eval(pipeline)
    assert not fitness.valid
