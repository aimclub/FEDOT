import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def without_diff_pipeline():
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_ridge)


def with_diff_pipeline():
    node_diff = PrimaryNode('hvatov_filter')
    node_lagged1 = SecondaryNode('lagged', nodes_from=[node_diff])
    node_lagged2 = PrimaryNode('lagged')
    node_ridge2 = SecondaryNode('ridge', nodes_from=[node_lagged2, node_lagged1])
    return Pipeline(node_ridge2)


def run_experiment_with_diff(time_series, len_forecast=250):
    """ Function with example how time series differentiation could help to do more precise predictions.

    :param time_series: time series for prediction
    :param raw_model: if only raw model used
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))
    idx = pd.to_datetime(time_series["datetime"].values)
    time_series = time_series["value"].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)

    pipeline_diff = with_diff_pipeline()
    pipeline_no_diff = without_diff_pipeline()
    pipeline_diff.fit(train_data)
    pipeline_no_diff.fit(train_data)

    prediction_diff = pipeline_diff.predict(test_data)
    predict_diff = np.ravel(np.array(prediction_diff.predict))

    prediction_no_diff = pipeline_no_diff.predict(test_data)
    predict_no_diff = np.ravel(np.array(prediction_no_diff.predict))

    test_target = np.ravel(test_data.target)
    rmse_diff = mean_squared_error(test_target, predict_diff, squared=False)
    mae_diff = mean_absolute_error(test_target, predict_diff)
    print(f'RMSE with diff- {rmse_diff:.4f}')
    print(f'MAE with diff- {mae_diff:.4f}\n')

    rmse_no_diff = mean_squared_error(test_target, predict_no_diff, squared=False)
    mae_no_diff = mean_absolute_error(test_target, predict_no_diff)
    print(f'RMSE without diff- {rmse_no_diff:.4f}')
    print(f'MAE without diff- {mae_no_diff:.4f}\n')

    plt.plot(idx, time_series, label='Actual time series')
    plt.plot(prediction_diff.idx, predict_diff, label='Forecast with diff')
    plt.plot(prediction_no_diff.idx, predict_no_diff, label='Forecast without diff')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    time_series = pd.read_csv('../data/stackoverflow.csv')
    run_experiment_with_diff(time_series,
                             len_forecast=30)
