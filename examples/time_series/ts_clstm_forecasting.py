import json
import os

import numpy as np
import pandas as pd

from examples.pipeline_import_export import create_correct_path
from examples.time_series.ts_forecasting_composing import prepare_train_test_input, fit_predict_for_pipeline, \
    display_validation_metric
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
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
    file_path = os.path.join(project_root_path, 'examples/data/ts_long.csv')
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
    horizon = 24*2
    window_size = 29
    n_steps = 100
    (train_data, test_data), _ = get_ts_data_long(n_steps=n_steps + horizon, forecast_length=horizon)

    node_root = PrimaryNode("clstm")
    node_root.custom_params = {
        'window_size': window_size,
        'hidden_size': 135,
        'learning_rate': 0.0004,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 50
    }

    pipeline = Pipeline(node_root)
    pipeline.fit(train_data)
    prediction_before_export = pipeline.predict(test_data).predict[0]

    print(f'Before export {prediction_before_export[:4]}')

    # Export it
    pipeline_path = "clstm"
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline()
    new_pipeline.load(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = np.array(predicted_output_after_export.predict[0])

    print(f'After import {prediction_after_export[:4]}')

    dict_pipeline, dict_fitted_operations = pipeline.save()
    dict_pipeline = json.loads(dict_pipeline)
    pipeline_from_dict = Pipeline()
    pipeline_from_dict.load(dict_pipeline, dict_fitted_operations)

    predicted_output = pipeline_from_dict.predict(test_data)
    prediction = np.array(predicted_output.predict[0])
    print(f'Prediction from pipeline loaded from dict {prediction[:4]}')

    display_validation_metric(
        np.ravel(prediction_before_export),
        test_data.target, np.concatenate([test_data.features[-window_size:], test_data.target]),
        True)


def get_source_pipeline_clstm():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    clstm - - - - /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_clstm = PrimaryNode('clstm')
    node_clstm.custom_params = {
        'window_size': 29,
        'hidden_size': 50,
        'learning_rate': 0.004,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 3
    }
    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_clstm])
    pipeline = Pipeline(node_final)

    return pipeline


def run_ts_forecasting_problem(forecast_length=50,
                               with_visualisation=True) -> None:
    """ Function launch time series task with composing

    :param forecast_length: length of the forecast
    :param with_visualisation: is it needed to show the plots
    """
    file_path = '../../cases/data/metocean/metocean_data_test.csv'

    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    # Train/test split
    train_part = time_series[len(time_series)-200:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part,
                                                                forecast_length)

    # Get pipeline with pre-defined structure
    init_pipeline = get_source_pipeline_clstm()

    # Init check
    preds = fit_predict_for_pipeline(pipeline=init_pipeline,
                                     train_input=train_input,
                                     predict_input=predict_input)

    display_validation_metric(predicted=preds,
                              real=test_part,
                              actual_values=time_series[-100:],
                              is_visualise=with_visualisation)


if __name__ == '__main__':
    run_ts_forecasting_problem(forecast_length=50,
                               with_visualisation=True)
