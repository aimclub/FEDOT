import os
from random import seed

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.tasks.test_forecasting import get_ts_data

seed(1)
np.random.seed(1)


@pytest.fixture()
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_regression.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


def get_simple_regr_pipeline():
    final = PrimaryNode(operation_type='xgbreg')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_regr_pipeline():
    node_scaling = PrimaryNode(operation_type='scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_scaling])
    final = SecondaryNode('xgbreg', nodes_from=[node_ridge, node_linear])
    pipeline = Pipeline(final)

    return pipeline


def get_simple_class_pipeline():
    final = PrimaryNode(operation_type='logit')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_class_pipeline():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    pipeline = get_complex_class_pipeline()

    custom_params = dict(C=10)

    pipeline.root_node.custom_params = custom_params
    pipeline.fit(data)
    params = pipeline.root_node.fitted_operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_pipeline_tuner_regression_correct(data_fixture, request):
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                      loss_function=mse,
                                                      loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_pipeline_tuner_classification_correct(data_fixture, request):
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=1)
        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                      loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_sequential_tuner_regression_correct(data_fixture, request):
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                        loss_function=mse,
                                                        loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_sequential_tuner_classification_correct(data_fixture, request):
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=2)
        tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                        loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_certain_node_tuning_regression_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                    node_index=0,
                                                    loss_function=mse)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_certain_node_tuning_classification_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                    node_index=0,
                                                    loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_ts_pipeline_with_stats_model():
    """ Tests PipelineTuner for time series forecasting task with AR model """
    train_data, test_data = get_ts_data(n_steps=200, forecast_length=5)

    ar_pipeline = Pipeline(PrimaryNode('ar'))

    # Tune AR model
    tuner_ar = PipelineTuner(pipeline=ar_pipeline, task=train_data.task, iterations=3)
    tuned_ar_pipeline = tuner_ar.tune_pipeline(input_data=train_data,
                                               loss_function=mse)

    is_tuning_finished = True

    assert is_tuning_finished
