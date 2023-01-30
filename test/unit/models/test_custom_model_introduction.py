import logging
import os
import shutil
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from examples.advanced.time_series_forecasting.custom_model_tuning import get_fitting_custom_pipeline
from examples.simple.pipeline_import_export import create_correct_path
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def custom_model_imitation(_, idx, train_data, params):
    """
    Function imitates custom model behaviour
    :param train_data: np.array for training the model
    :param params: dict parameters for custom model calculation
    :return result: random np.array with the same shape as train_data,
                    but with linear transformation according to params
    """
    a = params.get('a')
    b = params.get('b')
    result = np.random.rand(*train_data.shape) * a + b
    out_type = 'ts'
    if len(train_data.shape) > 1:
        out_type = 'table'
    return result, out_type


def get_centered_pipeline(with_params=True) -> Pipeline:
    """
        lagged -> custom -> ridge
    """
    lagged_node = PipelineNode('lagged')
    custom_node = PipelineNode('custom', nodes_from=[lagged_node])
    if with_params:
        custom_node.parameters = {"a": -50,
                                     "b": 500,
                                     'model_predict': custom_model_imitation}

    node_final = PipelineNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)
    return pipeline


def get_starting_pipeline(with_params=True):
    """
        custom -> lagged -> ridge
    """

    custom_node = PipelineNode('custom')
    if with_params:
        custom_node.parameters = {"a": -50,
                                     "b": 500,
                                     'model_predict': custom_model_imitation}
    lagged_node = PipelineNode('lagged', nodes_from=[custom_node])
    node_final = PipelineNode('ridge', nodes_from=[lagged_node])
    pipeline = Pipeline(node_final)
    return pipeline


def get_input_data():
    test_file_path = str(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(test_file_path, '../../data/simple_sea_level.csv'))
    time_series = np.array(df['Level'])
    len_forecast = 50
    train_input, predict_input = \
        train_test_data_setup(InputData(idx=range(len(time_series)),
                                        features=time_series,
                                        target=time_series,
                                        task=Task(TaskTypesEnum.ts_forecasting,
                                                  TsForecastingParams(
                                                      forecast_length=len_forecast)),
                                        data_type=DataTypesEnum.ts))
    return train_input, predict_input


def prepare_data():
    test_file_path = str(os.path.dirname(__file__))
    df = np.asarray(pd.read_csv(os.path.join(test_file_path, '../../data/simple_sea_level.csv')))
    df = np.delete(df, 0, 1)
    df = df[0:100, :]
    t_arr = range(len(df))

    percent_train = 0.7
    n_train = round(percent_train * len(t_arr))

    hist = df

    forecast_length = len(t_arr) - n_train
    ds = {}

    task = Task(TaskTypesEnum.ts_forecasting,
                task_params=TsForecastingParams(forecast_length=forecast_length))

    idx = np.asarray(t_arr)

    for i in range(df.shape[1]):
        ds[f'data_source_ts/hist_{i}'] = InputData(idx=idx,
                                                   features=hist[:, i],
                                                   target=hist[:, 0],
                                                   data_type=DataTypesEnum.ts,
                                                   task=task)

        ds[f'data_source_ts/exog_{i}'] = InputData(idx=idx,
                                                   features=deepcopy(df[:, i]),
                                                   target=deepcopy(hist),
                                                   data_type=DataTypesEnum.ts,
                                                   task=task)

    input_data_train, input_data_test = train_test_data_setup(MultiModalData(ds))

    return input_data_train, input_data_test


def model_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    return object


def model_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    # there we can face with several variant due to mutations
    if len(features.shape) > 1:
        return features[:, 0], 'ts'
    return features, 'ts'


def get_simple_pipeline(multi_data):
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """

    hist_list = []
    exog_list = []

    for i, data_id in enumerate(multi_data.keys()):
        if 'exog_' in data_id:
            exog_list.append(PipelineNode(data_id))
        if 'hist_' in data_id:
            lagged_node = PipelineNode('lagged', nodes_from=[PipelineNode(data_id)])
            lagged_node.parameters = {'window_size': 1}

            hist_list.append(lagged_node)

    # For custom model params as initial approximation and model as function is necessary
    custom_node = PipelineNode('custom/empty', nodes_from=exog_list)
    custom_node.parameters = {'model_predict': model_predict,
                                 'model_fit': model_fit}

    exog_pred_node = PipelineNode('exog_ts', nodes_from=[custom_node])

    final_ens = [exog_pred_node] + hist_list
    node_final = PipelineNode('ridge', nodes_from=final_ens)
    pipeline = Pipeline(node_final)

    return pipeline


def test_pipeline_with_custom_node():
    train_input, predict_input = get_input_data()
    pipeline = get_centered_pipeline()
    pipeline.fit_from_scratch(train_input)
    predicted_centered = pipeline.predict(predict_input)

    train_input, predict_input = get_input_data()
    pipeline = get_starting_pipeline()
    pipeline.fit_from_scratch(train_input)
    predicted_starting = pipeline.predict(predict_input)

    assert predicted_centered and predicted_starting is not None


def test_pipeline_with_custom_fitted_node():
    train_input, predict_input = get_input_data()
    pipeline = get_fitting_custom_pipeline()
    pipeline.fit_from_scratch(train_input)
    predicted_centered = pipeline.predict(predict_input)

    assert predicted_centered is not None


def test_save_pipeline_with_custom():
    train_input, predict_input = get_input_data()

    pipeline = get_centered_pipeline()
    pipeline.fit_from_scratch(train_input)

    pipeline.save(path='test_pipeline', create_subdir=False)
    json_path_load = create_correct_path('test_pipeline')
    new_pipeline = Pipeline.from_serialized(json_path_load)
    predicted_output_after_export = new_pipeline.predict(predict_input)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    # recursive deleting
    dir_ = os.path.dirname(json_path_load)
    shutil.rmtree(dir_)

    assert prediction_after_export is not None


def test_advanced_pipeline_with_custom_model():
    train_data, test_data = prepare_data()

    pipeline = get_simple_pipeline(train_data)

    pipeline.fit_from_scratch(train_data)
    predicted_test = pipeline.predict(test_data)

    assert predicted_test is not None


def test_composing_with_custom_model():
    train_data, test_data = prepare_data()

    initial_assumption = get_simple_pipeline(train_data)
    automl = Fedot(problem='ts_forecasting',
                   timeout=0.1,
                   task_params=TsForecastingParams(forecast_length=5), logging_level=logging.ERROR,
                   initial_assumption=initial_assumption,
                   preset='ts')
    pipeline = automl.fit(train_data)

    pipeline.fit_from_scratch(train_data)
    predicted_test = pipeline.predict(test_data)

    assert predicted_test is not None
