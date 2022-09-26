import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
from examples.simple.pipeline_import_export import create_correct_path
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_ts_data_long(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'examples/data/ts/ts_long.csv')
    df = pd.read_csv(file_path)
    df = df[df["series_id"] == "traffic_volume"]
    time_series = np.array(df['value'])[:n_steps]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data), task


def clstm_forecasting():
    """ Example of clstm pipeline serialization """
    horizon = 24 * 2
    window_size = 29
    n_steps = 100
    (train_data, test_data), _ = get_ts_data_long(n_steps=n_steps + horizon, forecast_length=horizon)

    node_root = PrimaryNode("clstm")
    node_root.parameters = {'window_size': window_size,
                            'hidden_size': 135,
                            'learning_rate': 0.0004,
                            'cnn1_kernel_size': 5,
                            'cnn1_output_size': 32,
                            'cnn2_kernel_size': 4,
                            'cnn2_output_size': 32,
                            'batch_size': 64,
                            'num_epochs': 50}

    pipeline = Pipeline(node_root)
    pipeline.fit(train_data)
    prediction_before_export = pipeline.predict(test_data).predict[0]

    print(f'Before export {prediction_before_export[:4]}')

    # Export it
    pipeline_path = "clstm"
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = np.array(predicted_output_after_export.predict[0])

    print(f'After import {prediction_after_export[:4]}')

    dict_pipeline, dict_fitted_operations = pipeline.save()
    dict_pipeline = json.loads(dict_pipeline)
    pipeline_from_dict = Pipeline.from_serialized(dict_pipeline, dict_fitted_operations)

    predicted_output = pipeline_from_dict.predict(test_data)
    prediction = np.array(predicted_output.predict[0])
    print(f'Prediction from pipeline loaded from dict {prediction[:4]}')

    plot_info = [
        {'idx': np.arange(np.concatenate([test_data.features, test_data.target]).shape[0]),
         'series': np.concatenate([test_data.features, test_data.target]),
         'label': 'Actual time series'},
        {'idx': np.arange(test_data.idx[0], test_data.idx[0] + horizon),
         'series': np.ravel(prediction_before_export),
         'label': 'prediction'},
        get_border_line_info(np.arange(test_data.idx[0] + 1)[-1],
                             prediction,
                             np.ravel(np.concatenate([test_data.features, test_data.target])),
                             'Border line')]

    rmse = mean_squared_error(test_data.target, prediction_before_export, squared=False)
    mae = mean_absolute_error(test_data.target, prediction_before_export)
    print(f'RMSE - {rmse:.4f}')
    print(f'MAE - {mae:.4f}')

    visualise(plot_info)


if __name__ == '__main__':
    clstm_forecasting()
