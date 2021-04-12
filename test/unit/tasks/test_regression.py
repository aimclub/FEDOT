import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as mse

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def generate_chain() -> Chain:
    node_scaling = PrimaryNode('scaling')
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_root = SecondaryNode('linear', nodes_from=[node_lasso, node_ridge])
    chain = Chain(node_root)
    return chain


def get_synthetic_regression_data(n_samples=10000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.regression),
                           data_type=DataTypesEnum.table)
    return input_data


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target, y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target, y_pred=train_pred.predict, squared=False)

    return rmse_value_train, rmse_value_test


def test_regression_chain_fit_correct():
    data = get_synthetic_regression_data()

    chain = generate_chain()
    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target) * 0.05
    assert rmse_on_test < rmse_threshold


def test_regression_chain_with_data_operation_fit_correct():
    data = get_synthetic_regression_data()
    train_data, test_data = train_test_data_setup(data)

    #           linear
    #       /           \
    #     ridge          |
    #       |            |
    # ransac_lin_reg   lasso
    #        \         /
    #          scaling
    node_scaling = PrimaryNode('scaling')
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_ransac])
    node_root = SecondaryNode('linear', nodes_from=[node_lasso, node_ridge])
    chain = Chain(node_root)

    chain.fit(train_data)
    results = chain.predict(test_data)

    assert results.predict.shape == test_data.target.shape
