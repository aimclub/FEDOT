from functools import partial

import pytest

from fedot.core.log import default_log
from fedot.core.optimisers.objective import Objective, PipelineObjectiveEvaluate
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.validation.split import tabular_cv_generator, OneFoldInputDataSplit
from test.unit.validation.test_table_cv import sample_pipeline
from test.unit.models.test_model import classification_dataset


_ = classification_dataset


def empty_pipeline():
    return Pipeline()


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

