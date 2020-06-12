import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements, DummyChainTypeEnum, DummyComposer
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum


def compose_chain(data: InputData) -> Chain:
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=['lasso', 'ridge'],
                                                 secondary=['linear'])

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
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

    chain = compose_chain(data=data)
    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target) * 0.05
    assert rmse_on_test < rmse_threshold


def test_regression_chain_with_datamodel_fit_correct():
    data = get_synthetic_regression_data()
    train_data, test_data = train_test_data_setup(data)

    chain = Chain()
    node_data = NodeGenerator.primary_node('direct_datamodel')
    node_first = NodeGenerator.primary_node('ridge')
    node_second = NodeGenerator.secondary_node('lasso')
    node_second.nodes_from = [node_first, node_data]

    chain.add_node(node_data)
    chain.add_node(node_first)
    chain.add_node(node_second)

    chain.fit(train_data)
    results = chain.predict(test_data)

    assert results.predict.shape == test_data.target.shape
