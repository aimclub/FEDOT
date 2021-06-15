import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode

def prepare_data(path_to_file, path_to_exog_file, ts_name, len_forecast=250):
    df = pd.read_csv(path_to_file)
    time_series = np.array(df[ts_name])
    df = pd.read_csv(path_to_exog_file)
    exog_variable = np.array(df[ts_name])

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Nemo feature
    train_data_exog = exog_variable[:-len_forecast]
    test_data_exog = exog_variable[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast

    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    train_input_exog = InputData(idx=np.arange(0, len(train_data_exog)),
                                 features=train_data_exog,
                                 target=train_data,
                                 task=task,
                                 data_type=DataTypesEnum.ts)

    predict_input_exog = InputData(idx=np.arange(start_forecast, end_forecast),
                                   features=test_data_exog,
                                   target=None,
                                   task=task,
                                   data_type=DataTypesEnum.ts)

    input_data = np.vstack([train_input, train_input_exog, predict_input, predict_input_exog])
    return input_data, test_data

def get_arima_nemo_chain(input_data):
    """ Function return complex chain with the following structure
        scaling -> arima \
                          linear
        nemo -> scaling  |
    """

    node_scaling = PrimaryNode('scaling', node_data={'fit': input_data[0][0],
                                                     'predict': input_data[2][0]})

    node_arima = SecondaryNode('arima', nodes_from=[node_scaling])

    node_nemo = PrimaryNode('exog', node_data={'fit': input_data[1][0],
                                               'predict': input_data[3][0]})

    node_nemo_scaling = SecondaryNode('scaling', nodes_from=[node_nemo])

    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo_scaling])
    chain = Chain(node_final)
    return chain


def get_STLarima_nemo_chain(input_data):
    """ Function return complex chain with the following structure
        scaling -> stl_arima \
                             linear
            nemo -> scaling  |
    """

    node_scaling = PrimaryNode('scaling', node_data={'fit': input_data[0][0],
                                                   'predict': input_data[2][0]})

    node_arima = SecondaryNode('stl_arima', nodes_from=[node_scaling])

    node_nemo = PrimaryNode('exog', node_data={'fit': input_data[1][0],
                                               'predict': input_data[3][0]})

    node_nemo_scaling = SecondaryNode('scaling', nodes_from=[node_nemo])

    node_final = SecondaryNode('linear', nodes_from=[node_arima, node_nemo_scaling])
    chain = Chain(node_final)
    return chain


def get_ridge_nemo_chain(input_data):
    """ Function return complex chain with the following structure
        lagged -> ridge \
                          ridge
        lagged -> ridge  |      \
                                 linear
                          nemo  /
    """

    node_lagged_1 = PrimaryNode('lagged', node_data={'fit': input_data[0][0],
                                                     'predict': input_data[2][0]})

    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    node_lagged_2 = PrimaryNode('lagged', node_data={'fit': input_data[0][0],
                                                     'predict': input_data[2][0]})

    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_ridge_3 = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])

    node_nemo = PrimaryNode('exog', node_data={'fit': input_data[1][0],
                                               'predict': input_data[3][0]})

    node_final = SecondaryNode('linear', nodes_from=[node_ridge_3, node_nemo])
    chain = Chain(node_final)
    return chain


def get_arima_chain(input_data):
    """ Function return complex chain with the following structure
        scaling -> arima
    """

    node_scaling = PrimaryNode('scaling', node_data={'fit': input_data[0][0],
                                                   'predict': input_data[2][0]})

    node_final = SecondaryNode('arima', nodes_from=[node_scaling])
    chain = Chain(node_final)
    return chain


def get_STLarima_chain(input_data):
    """ Function return complex chain with the following structure
        scaling -> stl_arima
    """

    node_scaling = PrimaryNode('scaling', node_data={'fit': input_data[0][0],
                                                   'predict': input_data[2][0]})

    node_final = SecondaryNode('stl_arima', nodes_from=[node_scaling])
    chain = Chain(node_final)
    return chain

def get_ridge_chain(input_data):
    """ Function return complex chain with the following structure
        lagged -> ridge \
                          ridge
        lagged -> ridge  |
    """

    node_lagged_1 = PrimaryNode('lagged', node_data={'fit': input_data[0][0],
                                                     'predict': input_data[2][0]})

    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    node_lagged_2 = PrimaryNode('lagged', node_data={'fit': input_data[0][0],
                                                     'predict': input_data[2][0]})

    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)
    return chain


def compare_plot(predicted, real, forecast_length, model):
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('SSH, m')
    plt.title(f'Sea surface height forecast for {forecast_length} days with {model}')
    plt.show()


def run_nemo_based_forecasting(path_to_file, path_to_exog_file, ts_name, len_forecast=60, is_visualise=False):
    c_input, test_data = prepare_data(path_to_file=path_to_file,
                                      path_to_exog_file=path_to_exog_file,
                                      ts_name=ts_name,
                                      len_forecast=len_forecast)

    """ Arima models """

    # simple arima
    chain = get_arima_chain(c_input)
    chain.fit_from_scratch()
    predicted_values = chain.predict()
    predicted_values = predicted_values.predict

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'arima')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'ARIMA RMSE - {mse_before:.4f}')
    print(f'ARIMA MAE - {mae_before:.4f}\n')

    # arima with nemo ensemble
    chain = get_arima_nemo_chain(c_input)
    chain.fit_from_scratch()
    predicted_values = chain.predict()
    predicted_values = predicted_values.predict

    predicted = np.ravel(np.array(predicted_values))
    test_data = np.ravel(test_data)

    if is_visualise:
        compare_plot(predicted, test_data, len_forecast, 'ARIMA with nemo')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'ARIMA with nemo RMSE - {mse_before:.4f}')
    print(f'ARIMA with nemo MAE - {mae_before:.4f}\n')

if __name__ == '__main__':
    run_nemo_based_forecasting(path_to_file='../cases/data/nemo/sea_surface_height_nemo.csv',
                               path_to_exog_file='../cases/data/nemo/sea_surface_height.csv',
                               ts_name='sea_level',
                               len_forecast=40,
                               is_visualise=True)