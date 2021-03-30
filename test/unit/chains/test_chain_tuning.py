import os
from random import seed

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.chains.tuning.sequential import SequentialTuner

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


def get_regr_chain():
    final = PrimaryNode(operation_type='xgbreg')
    chain = Chain(final)

    return chain


def get_class_chain():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    chain = Chain(final)

    return chain


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    chain = get_class_chain()

    custom_params = dict(C=10)

    chain.root_node.custom_params = custom_params
    chain.fit(data)
    params = chain.root_node.cache.actual_cached_state.operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_hp_chain_tuner(data_fixture, request):
    """ Test ChainTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    chain = get_regr_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # Chain tuning
    chain_tuner = ChainTuner(chain=chain,
                             task=train_data.task,
                             iterations=30)
    tuned_chain = chain_tuner.tune_chain(input_data=train_data,
                                         loss_function=mse)

    # After tuning prediction
    tuned_chain.fit_from_scratch(train_data)
    after_tuning_predicted = tuned_chain.predict(test_data)

    # Metrics
    bfr_tun_mse = mse(y_true=test_data.target,
                      y_pred=before_tuning_predicted.predict)
    aft_tun_mse = mse(y_true=test_data.target,
                      y_pred=after_tuning_predicted.predict)

    print(f'Before tune test {bfr_tun_mse}')
    print(f'After tune test {aft_tun_mse}', '\n')

    assert aft_tun_mse <= bfr_tun_mse


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_hp_sequential_tuner(data_fixture, request):
    """ Test SequentialTuner for chain based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    chain = get_class_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # Chain tuning
    sequential_tuner = SequentialTuner(chain=chain,
                                       task=train_data.task,
                                       iterations=30)
    tuned_chain = sequential_tuner.tune_chain(input_data=train_data,
                                              loss_function=roc)

    # After tuning prediction
    tuned_chain.fit_from_scratch(train_data)
    after_tuning_predicted = tuned_chain.predict(test_data)

    # Metrics
    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 1)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tuning_predicted.predict), 1)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc
