import logging
import random
from copy import deepcopy

import pytest

from examples.simple.classification.classification_pipelines import (classification_pipeline_with_balancing,
                                                                     classification_pipeline_without_balancing)
from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.integration.api.test_main_api import get_dataset, get_cholesterol_dataset
from test.unit.tasks.test_classification import get_binary_classification_data


def test_compose_fedot_model_without_tuning():
    task_type = 'classification'
    train_input, _, _ = get_dataset(task_type=task_type)

    model = Fedot(problem=task_type, timeout=0.1, preset='fast_train', with_tuning=True)
    model.fit(train_input)

    assert not model.api_composer.was_tuned


def test_output_binary_classification_correct():
    """ Check the correctness of prediction for binary classification task """

    task_type = 'classification'

    data = get_binary_classification_data()

    random.seed(1)
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    model = Fedot(problem=task_type, timeout=0.1)
    model.fit(train_data, predefined_model='logit')
    model.predict(test_data)
    metrics = model.get_metrics(metric_names=['roc_auc', 'f1'])

    assert all(value >= 0.6 for value in metrics.values())


def test_predefined_initial_assumption():
    """ Check if predefined initial assumption and other api params don't lose while preprocessing is performing"""
    train_input, _, _ = get_dataset(task_type='classification')
    initial_pipelines = [classification_pipeline_without_balancing(), classification_pipeline_with_balancing()]
    available_operations = ['bernb', 'dt', 'knn', 'lda', 'qda', 'logit', 'rf', 'svc',
                            'scaling', 'normalization', 'pca', 'kernel_pca']

    model = Fedot(problem='classification', timeout=.1,
                  logging_level=logging.DEBUG, available_operations=available_operations,
                  initial_assumption=initial_pipelines)
    old_params = deepcopy(model.params)
    model.fit(train_input)

    assert len(initial_pipelines) == len(model.params.get('initial_assumption'))
    assert len(model.params.get('initial_assumption')) == len(model.history.initial_assumptions)
    assert len(old_params) == len(model.params)


@pytest.mark.parametrize('train_input', [
    get_dataset(task_type='regression')[0],
    get_cholesterol_dataset()[0],
])
def test_predefined_model_task_mismatch_is_caught(train_input):
    # task is regression, but model is for classification
    problem = 'regression'
    predefined_model = 'logit'
    model = Fedot(problem=problem)

    with pytest.raises(ValueError):
        model.fit(features=train_input, predefined_model=predefined_model)


def test_the_formation_of_initial_assumption():
    """ Checks that the initial assumption is formed based on the given available operations """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    available_operations = ['dt']

    initial_assumptions = AssumptionsBuilder \
        .get(train_input) \
        .from_operations(available_operations) \
        .build()
    res_init_assumption = Pipeline(PipelineNode('dt'))
    assert initial_assumptions[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_init_assumption_with_inappropriate_available_operations():
    """ Checks that if given available operations are not suitable for the task,
    then the default initial assumption will be formed """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    available_operations = ['linear', 'xgboost', 'lagged']

    initial_assumptions = AssumptionsBuilder \
        .get(train_input) \
        .from_operations(available_operations) \
        .build()
    primary = PipelineNode('scaling')
    root = PipelineNode('rf', nodes_from=[primary])
    res_init_assumption = Pipeline(root)

    assert initial_assumptions[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_api_composer_available_operations():
    """ Checks if available_operations goes through all fitting process"""
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=1))
    train_data, _, _ = get_dataset(task_type='ts_forecasting')
    available_operations = ['lagged']
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=0.01,
                  available_operations=available_operations,
                  pop_size=500
                  )
    model.fit(train_data)
    assert model.params.get('available_operations') == available_operations
