import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def get_composite_chain():
    """ Function return complex chain with the following structure
    lagged -> ridge \
                     ridge
    lagged -> treg  |
    """
    chain = Chain()
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.custom_params = {'window_size': 110}
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    node_lagged_2 = PrimaryNode('lagged')
    node_lagged_2.custom_params = {'window_size': 20}
    node_treg = SecondaryNode('treg', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('ridge', nodes_from=[node_treg, node_ridge])
    chain.add_node(node_final)
    return chain


def get_simple_chain():
    """ Function return simple chain with the following structure
    lagged -> linear
    """
    node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    chain_simple = Chain(node_final)

    return chain_simple


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, is_visualise=False) -> float:
    """ Function for calculate RMSE metric

    :param pred: predicted OutputData
    :param valid: test data for validation
    :param name: name of the model for visualisation
    :param is_visualise: is it need to visualise

    :return rmse: RMSE metric value
    """
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = np.ravel(pred.predict)
    real = np.ravel(valid.target)

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    rmse = mse(y_true=real,
               y_pred=predicted,
               squared=False)

    return rmse


def compare_plot(predicted, real, forecast_length, model_name):
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('SSH, cm')
    plt.title(f'Sea surface height forecast for {forecast_length} hours with {model_name}')
    plt.show()


def prepare_input_data(train_file_path, test_file_path, forecast_length):
    """ Function for preparing InputData for train and test algorithm

    :param train_file_path: path to the csv file for training
    :param test_file_path: path to the csv file for validation
    :param forecast_length: forecast length for prediction

    :return dataset_to_train: InputData for train
    :return dataset_to_validate: InputData for validation
    """
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length))

    # Load train and test dataframes
    full_path_train = os.path.join(str(project_root()), train_file_path)
    full_path_test = os.path.join(str(project_root()), test_file_path)
    df_train = pd.read_csv(full_path_train)
    df_test = pd.read_csv(full_path_test)

    # Get idx for train and series for train
    train_feature_ts = np.ravel(np.array(df_train['wind_speed']))
    train_target_ts = np.ravel(np.array(df_train['sea_height']))
    idx_train = np.arange(0, len(train_feature_ts))
    dataset_to_train = InputData(idx=idx_train,
                                 features=train_feature_ts,
                                 target=train_target_ts,
                                 task=task_to_solve,
                                 data_type=DataTypesEnum.ts)

    start_forecast = len(idx_train)
    end_forecast = start_forecast + forecast_length
    idx_test = np.arange(start_forecast, end_forecast)

    test_target_ts = np.ravel(np.array(df_test['sea_height']))
    test_target_ts = test_target_ts[:forecast_length]
    dataset_to_validate = InputData(idx=idx_test,
                                    features=train_feature_ts,
                                    target=test_target_ts,
                                    task=task_to_solve,
                                    data_type=DataTypesEnum.ts)

    return dataset_to_train, dataset_to_validate


def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=1, is_visualise=False,
                                     with_composite_chain=False):
    # Prepare data for train and test
    dataset_to_train, dataset_to_validate = prepare_input_data(train_file_path,
                                                               test_file_path,
                                                               forecast_length)

    # Simple chain
    chain_simple = get_simple_chain()

    chain_simple.fit_from_scratch(input_data=dataset_to_train)
    rmse_on_valid_simple = calculate_validation_metric(
        chain_simple.predict(dataset_to_validate), dataset_to_validate,
        f'full-simple_{forecast_length}',
        is_visualise=is_visualise)
    print(f'RMSE simple: {rmse_on_valid_simple}')

    if with_composite_chain:
        chain_composite = get_composite_chain()
        chain_composite.fit(input_data=dataset_to_train)
        rmse_on_valid_lstm_only = calculate_validation_metric(
            chain_composite.predict(dataset_to_validate), dataset_to_validate,
            f'full-composite_{forecast_length}',
            is_visualise=is_visualise)
        print(f'RMSE composite: {rmse_on_valid_lstm_only}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test,
                                     forecast_length=72,
                                     is_visualise=True,
                                     with_composite_chain=True)
