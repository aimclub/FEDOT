import os

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from examples.pipeline_import_export import create_correct_path
from examples.time_series.ts_custom_model_tuning import get_fitting_custom_pipeline


def custom_model_imitation(_, train_data, params):
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
    lagged_node = PrimaryNode('lagged')
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    if with_params:
        custom_node.custom_params = {"a": -50,
                                     "b": 500,
                                     'model_predict': custom_model_imitation}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)
    return pipeline


def get_starting_pipeline(with_params=True):
    """
        custom -> lagged -> ridge
    """

    custom_node = PrimaryNode('custom')
    if with_params:
        custom_node.custom_params = {"a": -50,
                                     "b": 500,
                                     'model_predict': custom_model_imitation}
    lagged_node = SecondaryNode('lagged', nodes_from=[custom_node])
    node_final = SecondaryNode('ridge', nodes_from=[lagged_node])
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

    pipeline.save(path='test_pipeline')
    json_path_load = create_correct_path('test_pipeline')
    new_pipeline = Pipeline()
    new_pipeline.load(json_path_load)
    predicted_output_after_export = new_pipeline.predict(predict_input)
    prediction_after_export = np.array(predicted_output_after_export.predict)
    os.remove(json_path_load)

    assert prediction_after_export is not None
