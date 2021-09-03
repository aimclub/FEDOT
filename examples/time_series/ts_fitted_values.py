import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import fitted_values
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from examples.time_series.ts_forecasting_tuning import get_complex_pipeline


def show_fitted_time_series(len_forecast=24):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    ts_input = InputData.from_csv_time_series(file_path='../../cases/data/time_series/metocean.csv',
                                              task=task, target_column='value')

    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 4}
    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    train_predicted = pipeline.fit(ts_input)

    fitted_ts_10 = fitted_values(train_predicted, 10)
    fitted_ts_act = fitted_values(train_predicted)
    plt.plot(ts_input.idx, ts_input.target, label='Actual time series')
    plt.plot(fitted_ts_10.idx, fitted_ts_10.predict, label='Fitted values horizon 10')
    plt.plot(fitted_ts_act.idx, fitted_ts_act.predict, label='Fitted values all')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    show_fitted_time_series()
