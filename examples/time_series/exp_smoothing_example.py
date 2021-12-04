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


def ets_pipeline():
    node_cut = PrimaryNode('cut')
    node_ets = SecondaryNode('ets', nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              'seasonal': "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}
    return Pipeline(node_ets)


def ets_ridge_pipeline():
    node_cut = PrimaryNode('cut')
    node_cut.custom_params = {"cut_part": 0.5}
    node_ets = SecondaryNode('ets', nodes_from=[node_cut])
    node_ets.custom_params = {"error": "add",
                              "trend": "add",
                              'seasonal': "add",
                              "damped_trend": False,
                              "seasonal_periods": 20}
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_ridge1 = SecondaryNode('ridge', nodes_from=[node_ridge, node_ets])

    return Pipeline(node_ridge1)


def run_experiment_with_ets(time_series, raw_model=False, len_forecast=250):
    """ Function with example of ts forecasting using exponential smoothing
    :param time_series: time series for prediction
    :param raw_model: use raw ets model or ets+ridge
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))
    idx = pd.to_datetime(time_series["Month"].values)
    time_series = time_series["Monthly beer production"].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)

    if raw_model:
        pipeline = ets_pipeline()
    else:
        pipeline = ets_ridge_pipeline()
    pipeline.fit(train_data)

    prediction_before = pipeline.predict(test_data)
    predict_before = np.ravel(np.array(prediction_before.predict))
    test_target = np.ravel(test_data.target)

    rmse_before = mean_squared_error(test_target, predict_before, squared=False)
    mae_before = mean_absolute_error(test_target, predict_before)
    print(f'RMSE before tuning - {rmse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    plt.plot(idx, time_series, label='Actual time series')
    plt.plot(prediction_before.idx, predict_before, label='Forecast before tuning')

    pipeline = pipeline.fine_tune_all_nodes(input_data=train_input,
                                            loss_function=mean_squared_error,
                                            loss_params={'squared': False},
                                            )

    prediction_after = pipeline.predict(test_data)
    predict_after = np.ravel(np.array(prediction_after.predict))

    rmse_after = mean_squared_error(test_target, predict_after, squared=False)
    mae_after = mean_absolute_error(test_target, predict_after)
    print(f'RMSE after tuning - {rmse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    plt.plot(prediction_after.idx, predict_after, label='Forecast after tuning')
    plt.legend()
    plt.grid()
    plt.show()

    pipeline.print_structure()


if __name__ == '__main__':
    time_series = pd.read_csv('../data/beer.csv')
    run_experiment_with_ets(time_series,
                            raw_model=False,
                            len_forecast=50)
