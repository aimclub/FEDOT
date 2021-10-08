import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast_with_tuning(pipeline, train_input, predict_input, task, cv_folds):
    """
    Function for predicting values in a time series

    :param pipeline: TsForecastingPipeline object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task
    :param cv_folds: number of folds for validation

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    pipeline.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Predict
    predicted_values = pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict

    pipeline_tuner = PipelineTuner(pipeline=pipeline, task=task,
                                   iterations=10)
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                            loss_function=mean_squared_error,
                                            loss_params={'squared': False},
                                            cv_folds=cv_folds,
                                            validation_blocks=3)

    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict
    predicted_values = pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values


def get_complex_pipeline():
    """
    Pipeline looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /
    """

    # First level
    node_smoothing = PrimaryNode('smoothing')

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_2 = PrimaryNode('lagged')

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def get_ar_pipeline():
    """
    Function return graph with AR model
    """

    node_ar = PrimaryNode('ar')
    pipeline = Pipeline(node_ar)

    return pipeline


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Return prepared data for fit and predict

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


def run_experiment_with_tuning(time_series, with_ar_pipeline=False, len_forecast=250,
                               cv_folds=None):
    """ Function with example how time series forecasting can be made

    :param time_series: time series for prediction
    :param with_ar_pipeline: is it needed to use pipeline with AR model or not
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    # Get graph with several models and with arima pipeline
    if with_ar_pipeline:
        pipeline = get_ar_pipeline()
    else:
        pipeline = get_complex_pipeline()

    old_predicted, new_predicted = make_forecast_with_tuning(pipeline, train_input,
                                                             predict_input, task,
                                                             cv_folds)

    old_predicted = np.ravel(np.array(old_predicted))
    new_predicted = np.ravel(np.array(new_predicted))
    test_data = np.ravel(test_data)

    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    pipeline.print_structure()
    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), old_predicted, label='Forecast before tuning')
    plt.plot(range(len(train_data), len(time_series)), new_predicted, label='Forecast after tuning')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../../cases/data/time_series/metocean.csv')
    time_series = np.array(df['value'])
    run_experiment_with_tuning(time_series,
                               with_ar_pipeline=False,
                               len_forecast=50,
                               cv_folds=2)
