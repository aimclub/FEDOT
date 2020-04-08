import numpy as np
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_process import ArmaProcess

from core.composer.chain import Chain
from core.composer.composer import DummyComposer, DummyChainTypeEnum, ComposerRequirements
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum


def compose_chain(data: InputData) -> Chain:
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.arima, ModelTypesIdsEnum.arima],
                                                 secondary=[ModelTypesIdsEnum.linear])

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
    return chain


def get_synthetic_ts_data(n_steps=10000) -> InputData:
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data = simulated_data + x1 * 0.0005 - x2 * 0.0001

    input_data = InputData(idx=np.arange(0, n_steps),
                           features=np.asarray([x1, x2]).T,
                           target=simulated_data,
                           task_type=MachineLearningTasksEnum.auto_regression)
    return input_data


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target, y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target, y_pred=train_pred.predict, squared=False)

    return rmse_value_train, rmse_value_test


def test_regression_chain_fit_correct():
    data = get_synthetic_ts_data()

    chain = compose_chain(data=data)
    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target)

    assert rmse_on_test < rmse_threshold
