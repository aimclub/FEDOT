import os
from random import seed

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.sequential import SequentialTuner
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.tasks.test_forecasting import get_synthetic_ts_data_period

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


def get_simple_regr_chain():
    final = PrimaryNode(operation_type='xgbreg')
    chain = Chain(final)

    return chain


def get_complex_regr_chain():
    node_scaling = PrimaryNode(operation_type='scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_scaling])
    final = SecondaryNode('xgbreg', nodes_from=[node_ridge, node_linear])
    chain = Chain(final)

    return chain


def get_simple_class_chain():
    final = PrimaryNode(operation_type='logit')
    chain = Chain(final)

    return chain


def get_complex_class_chain():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    chain = get_complex_class_chain()

    custom_params = dict(C=10)

    chain.root_node.custom_params = custom_params
    chain.fit(data)
    params = chain.root_node.fitted_operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_chain_tuner_regression_correct(data_fixture, request):
    """ Test ChainTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for regression task
    chain_simple = get_simple_regr_chain()
    chain_complex = get_complex_regr_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        chain_tuner = ChainTuner(chain=chain,
                                 task=train_data.task,
                                 iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_chain = chain_tuner.tune_chain(input_data=train_data,
                                             loss_function=mse,
                                             loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_chain_tuner_classification_correct(data_fixture, request):
    """ Test ChainTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for classification task
    chain_simple = get_simple_class_chain()
    chain_complex = get_complex_class_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        chain_tuner = ChainTuner(chain=chain,
                                 task=train_data.task,
                                 iterations=1)
        tuned_chain = chain_tuner.tune_chain(input_data=train_data,
                                             loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_sequential_tuner_regression_correct(data_fixture, request):
    """ Test SequentialTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for regression task
    chain_simple = get_simple_regr_chain()
    chain_complex = get_complex_regr_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        sequential_tuner = SequentialTuner(chain=chain,
                                           task=train_data.task,
                                           iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_chain = sequential_tuner.tune_chain(input_data=train_data,
                                                  loss_function=mse,
                                                  loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_sequential_tuner_classification_correct(data_fixture, request):
    """ Test SequentialTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for classification task
    chain_simple = get_simple_class_chain()
    chain_complex = get_complex_class_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        sequential_tuner = SequentialTuner(chain=chain,
                                           task=train_data.task,
                                           iterations=2)
        tuned_chain = sequential_tuner.tune_chain(input_data=train_data,
                                                  loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_certain_node_tuning_regression_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for regression task
    chain_simple = get_simple_regr_chain()
    chain_complex = get_complex_regr_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        sequential_tuner = SequentialTuner(chain=chain,
                                           task=train_data.task,
                                           iterations=1)
        tuned_chain = sequential_tuner.tune_node(input_data=train_data,
                                                 node_index=0,
                                                 loss_function=mse)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_certain_node_tuning_classification_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chains for classification task
    chain_simple = get_simple_class_chain()
    chain_complex = get_complex_class_chain()

    for chain in [chain_simple, chain_complex]:
        # Chain tuning
        sequential_tuner = SequentialTuner(chain=chain,
                                           task=train_data.task,
                                           iterations=1)
        tuned_chain = sequential_tuner.tune_node(input_data=train_data,
                                                 node_index=0,
                                                 loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_ts_chain_with_stats_model():
    """ Tests ChainTuner for time series forecasting task with AR model """
    train_data, test_data = get_synthetic_ts_data_period()

    ar_chain = Chain(PrimaryNode('ar'))

    # Tune AR model
    tuner_ar = ChainTuner(chain=ar_chain, task=train_data.task, iterations=5)
    tuned_ar_chain = tuner_ar.tune_chain(input_data=train_data,
                                         loss_function=mse)

    is_tuning_finished = True

    assert is_tuning_finished
