from datetime import timedelta
from typing import Optional, Union, Iterable

import numpy as np
import pytest
from golem.core.optimisers.fitness import Fitness
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.tuner_interface import HyperoptTuner
from hyperopt import tpe, rand

from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective.data_objective_eval import get_pipeline_evaluator
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricType
from test.unit.optimizer.test_pipeline_objective_eval import pipeline_first_test
from test.unit.pipelines.tuning.test_pipeline_tuning import get_not_default_search_space
from test.unit.validation.test_table_cv import get_classification_data


def test_tuner_builder_with_default_params():
    data = get_classification_data()
    pipeline = pipeline_first_test()
    tuner = TunerBuilder(data.task).build(data)
    objective_evaluate = get_pipeline_evaluator(ClassificationMetricsEnum.ROCAUC_penalty, data)
    assert isinstance(tuner, HyperoptTuner)
    assert np.isclose(tuner.objective_evaluate(pipeline).value, objective_evaluate.evaluate(pipeline).value)
    assert isinstance(tuner.search_space, PipelineSearchSpace)
    assert tuner.iterations == DEFAULT_TUNING_ITERATIONS_NUMBER
    assert tuner.algo == tpe.suggest
    assert tuner.max_seconds == 300


@pytest.mark.parametrize('tuner_class', [SimultaneousTuner, SequentialTuner])
def test_tuner_builder_with_custom_params(tuner_class):
    data = get_classification_data()
    pipeline = pipeline_first_test()
    metric = ClassificationMetricsEnum.accuracy
    cv_folds = 3
    validation_blocks = 2

    objective_evaluate = get_pipeline_evaluator(metric, data, cv_folds, validation_blocks)
    timeout = timedelta(minutes=2)
    early_stopping = 100
    iterations = 10
    algo = rand.suggest
    search_space = get_not_default_search_space()

    tuner = TunerBuilder(data.task)\
        .with_tuner(tuner_class)\
        .with_metric(metric)\
        .with_cv_folds(cv_folds)\
        .with_validation_blocks(validation_blocks)\
        .with_timeout(timeout)\
        .with_early_stopping_rounds(early_stopping)\
        .with_iterations(iterations)\
        .with_algo(algo)\
        .with_search_space(search_space)\
        .build(data)

    assert isinstance(tuner, tuner_class)
    assert np.isclose(tuner.objective_evaluate(pipeline).value, objective_evaluate.evaluate(pipeline).value)
    assert tuner.search_space == search_space
    assert tuner.iterations == iterations
    assert tuner.algo == algo
    assert tuner.max_seconds == int(timeout.seconds)
