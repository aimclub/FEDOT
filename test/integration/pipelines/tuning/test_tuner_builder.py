from datetime import timedelta
from typing import Optional

import numpy as np
import pytest
from golem.core.tuning.hyperopt_tuner import HyperoptTuner
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from hyperopt import tpe, rand

from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricType
from test.integration.pipelines.tuning.test_pipeline_tuning import get_not_default_search_space
from test.unit.optimizer.test_pipeline_objective_eval import pipeline_first_test
from test.unit.validation.test_table_cv import get_classification_data


def get_objective_evaluate(metric: MetricType, data: InputData,
                           cv_folds: Optional[int] = None, validation_blocks: Optional[int] = None) \
        -> PipelineObjectiveEvaluate:
    objective = MetricsObjective(metric)
    data_producer = DataSourceSplitter(cv_folds, validation_blocks).build(data)
    objective_evaluate = PipelineObjectiveEvaluate(objective, data_producer, validation_blocks=validation_blocks)
    return objective_evaluate


def test_tuner_builder_with_default_params():
    data = get_classification_data()
    pipeline = pipeline_first_test()
    tuner = TunerBuilder(data.task).build(data)
    objective_evaluate = get_objective_evaluate(ClassificationMetricsEnum.ROCAUC_penalty, data)
    assert isinstance(tuner, HyperoptTuner)
    assert np.isclose(tuner.objective_evaluate(pipeline).value, objective_evaluate.evaluate(pipeline).value)
    assert isinstance(tuner.search_space, PipelineSearchSpace)
    assert tuner.iterations == DEFAULT_TUNING_ITERATIONS_NUMBER
    assert tuner.algo == tpe.suggest
    assert tuner.max_seconds == 300


@pytest.mark.parametrize('tuner_class', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_tuner_builder_with_custom_params(tuner_class):
    data = get_classification_data()
    pipeline = pipeline_first_test()
    metric = ClassificationMetricsEnum.accuracy
    cv_folds = 3
    validation_blocks = 2

    objective_evaluate = get_objective_evaluate(metric, data, cv_folds, validation_blocks)
    timeout = timedelta(minutes=2)
    early_stopping = 100
    iterations = 10
    search_space = get_not_default_search_space()

    tuner = (
        TunerBuilder(data.task)
        .with_tuner(tuner_class)
        .with_metric(metric)
        .with_cv_folds(cv_folds)
        .with_validation_blocks(validation_blocks)
        .with_timeout(timeout)
        .with_early_stopping_rounds(early_stopping)
        .with_iterations(iterations)
        .with_search_space(search_space)
        .build(data)
    )

    assert isinstance(tuner, tuner_class)
    assert np.isclose(tuner.objective_evaluate(pipeline).value, objective_evaluate.evaluate(pipeline).value)
    assert tuner.search_space == search_space
    assert tuner.iterations == iterations
    assert tuner.max_seconds == int(timeout.seconds)
