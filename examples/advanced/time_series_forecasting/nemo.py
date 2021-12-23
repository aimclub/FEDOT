import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from copy import deepcopy
warnings.filterwarnings('ignore')


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features,
                            target=train_data_target,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=test_data_features,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def get_arima_pipeline():
    """ Function return complex pipeline with the following structure
        arima
    """

    node_1 = PrimaryNode('arima')
    node_final = SecondaryNode('linear', nodes_from=[node_1])

    pipeline = Pipeline(node_final)
    return pipeline


def get_arima_nemo_pipeline():
    """ Function return complex pipeline with the following structure
        arima \
               linear
        nemo  |
    """

    node_arima = PrimaryNode('arima')
    node_nemo = PrimaryNode('exog_ts')
    node_final = SecondaryNode('ridge', nodes_from=[node_arima, node_nemo])
    pipeline = Pipeline(node_final)
    return pipeline


def return_working_pipeline():
    node_lagged_1 = PrimaryNode('lagged/1')
    node_exog = PrimaryNode('exog_ts')

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged_1, node_exog])
    pipeline = Pipeline(node_final)
    return pipeline


len_forecast = 40
ts_name = 'sea_level'
path_to_file = '../../cases/data/nemo/sea_surface_height.csv'
path_to_exog_file = '../../cases/data/nemo/sea_surface_height_nemo.csv'

df = pd.read_csv(path_to_file)
time_series = np.array(df[ts_name])
df = pd.read_csv(path_to_exog_file)
exog_variable = np.array(df[ts_name])


# Let's divide our data on train and test samples
train_data = time_series[:-len_forecast]
test_data = time_series[-len_forecast:]

# Nemo features
train_data_exog = exog_variable[:-len_forecast]
test_data_exog = exog_variable[-len_forecast:]

# Source time series
train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                      train_data_features=train_data,
                                                      train_data_target=train_data,
                                                      test_data_features=train_data)

# Exogenous time series
train_input_exog, predict_input_exog, _ = prepare_input_data(len_forecast=len_forecast,
                                                             train_data_features=train_data_exog,
                                                             train_data_target=train_data,
                                                             test_data_features=test_data_exog)

pipeline = get_arima_pipeline()
train_dataset = MultiModalData({
        'arima': deepcopy(train_input),
    })
predict_dataset = MultiModalData({
        'arima': deepcopy(predict_input),
    })
pipeline.fit_from_scratch(train_dataset)
predicted_values = pipeline.predict(predict_dataset)
predicted_values = predicted_values.predict

predicted = np.ravel(np.array(predicted_values))
test_data = np.ravel(np.array(test_data))

mse_before = mean_squared_error(test_data, predicted, squared=False)
mae_before = mean_absolute_error(test_data, predicted)
mape_before = mean_absolute_percentage_error(test_data, predicted)
print(f'ARIMA MSE - {mse_before:.4f}')
print(f'ARIMA MAE - {mae_before:.4f}')
print(f'ARIMA MAPE - {mape_before:.4f}\n')


# arima with nemo ensemble
pipeline = return_working_pipeline()
train_dataset = MultiModalData({
        'lagged/1': deepcopy(train_input),
        'exog_ts': deepcopy(train_input_exog)
    })
predict_dataset = MultiModalData({
        'lagged/1': deepcopy(predict_input),
        'exog_ts': deepcopy(predict_input_exog)
    })
pipeline.fit_from_scratch(train_dataset)
predicted_values = pipeline.predict(predict_dataset).predict

predicted = np.ravel(np.array(predicted_values))
test_data = np.ravel(np.array(test_data))


mse_before = mean_squared_error(test_data, predicted, squared=False)
mae_before = mean_absolute_error(test_data, predicted)
mape_before = mean_absolute_percentage_error(test_data, predicted)
print(f'Lagged with nemo MSE - {mse_before:.4f}')
print(f'Lagged with nemo MAE - {mae_before:.4f}')
print(f'Lagged with nemo MAPE - {mape_before:.4f}\n')


# arima with nemo ensemble
pipeline = get_arima_nemo_pipeline()
train_dataset = MultiModalData({
        'arima': deepcopy(train_input),
        'exog_ts': deepcopy(train_input_exog)
    })
predict_dataset = MultiModalData({
        'arima': deepcopy(predict_input),
        'exog_ts': deepcopy(predict_input_exog)
    })
pipeline.fit_from_scratch(train_dataset)
predicted_values = pipeline.predict(predict_dataset).predict

predicted = np.ravel(np.array(predicted_values))
test_data = np.ravel(np.array(test_data))


mse_before = mean_squared_error(test_data, predicted, squared=False)
mae_before = mean_absolute_error(test_data, predicted)
mape_before = mean_absolute_percentage_error(test_data, predicted)
print(f'ARIMA with nemo MSE - {mse_before:.4f}')
print(f'ARIMA with nemo MAE - {mae_before:.4f}')
print(f'ARIMA with nemo MAPE - {mape_before:.4f}')
