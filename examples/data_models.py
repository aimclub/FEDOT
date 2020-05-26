import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc_auc
from statsmodels.tsa.arima_process import ArmaProcess

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, OutputData, train_test_data_setup
from core.models.model import *
from core.repository.model_types_repository import ModelMetaInfoTemplate, ModelTypesIdsEnum, ModelTypesRepository
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root


def get_synthetic_ts_data(n_steps=10000) -> InputData:
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps, scale=0.01)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data += np.sin([_ / 20 for _ in range(n_steps)])

    # simulated_data += simulated_data + x1 * 0.0005 - x2 * 0.0001

    simulated_data += np.sin([_ / 5 for _ in range(n_steps)]) / 4

    input_data = InputData(idx=np.arange(0, n_steps),
                           features=np.asarray([x1, x2]).T,
                           target=simulated_data,
                           task_type=MachineLearningTasksEnum.auto_regression)
    return input_data


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target, y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target, y_pred=train_pred.predict, squared=False)

    return rmse_value_train, rmse_value_test, train_pred, test_pred


def regression_chain_fit():
    data = get_synthetic_ts_data()

    train_data, test_data = train_test_data_setup(data)

    chain_simple = Chain()
    chain_simple.add_node(NodeGenerator.primary_node(ModelTypesIdsEnum.arima))

    chain = Chain()
    node0 = NodeGenerator.primary_node(ModelTypesIdsEnum.arima)

    node1 = NodeGenerator.secondary_node(ModelTypesIdsEnum.diff_data_model)
    node2 = NodeGenerator.secondary_node(ModelTypesIdsEnum.arima)
    node3 = NodeGenerator.secondary_node(ModelTypesIdsEnum.add_data_model)

    node1.nodes_from.append(node0)
    node2.nodes_from.append(node1)

    node3.nodes_from.append(node2)
    node3.nodes_from.append(node0)

    chain.add_node(node0)
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_node(node3)

    chain.fit(input_data=train_data)
    chain_simple.fit(input_data=train_data)

    rmse_on_train, rmse_on_test, train_pred, test_pred = get_rmse_value(chain, train_data, test_data)
    rmse_on_train_simple, rmse_on_test_simple, train_pred_simple, test_pred_simple = \
        get_rmse_value(chain_simple, train_data, test_data)

    print(rmse_on_train)
    print(rmse_on_train_simple)

    print(rmse_on_test)
    print(rmse_on_test_simple)

    compare_plot(train_pred, train_data)
    compare_plot(test_pred, test_data)
    compare_plot(train_pred_simple, train_data)
    compare_plot(test_pred_simple, test_data)


def compare_plot(predicted: OutputData, dataset_to_validate: InputData):
    fig, ax = plt.subplots()
    plt.plot(dataset_to_validate.target, linewidth=1, label="Observed")
    plt.plot(predicted.predict, linewidth=1, label="Predicted")
    ax.legend()

    plt.show()


regression_chain_fit()
