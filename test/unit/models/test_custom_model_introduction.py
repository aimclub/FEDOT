import numpy as np
import pandas as pd
import os

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from examples.time_series.ts_custom_model_tuning import prepare_input_data
from examples.pipeline_import_export import create_correct_path


def custom_model_imitation(train_data, _, params):
    """
    Function imitates custom model behaviour
    :param train_data: np.array for training the model
    :param params: dict parameters for custom model calculation
    :return result: random np.array with the same shape as train_data,
                    but with linear transformation according to params
    """
    a = params.get('a')
    b = params.get('b')
    result = np.random.rand(*train_data.shape)*a + b
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
                                     'model': custom_model_imitation}

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
                                     'model': custom_model_imitation}
    lagged_node = SecondaryNode('lagged', nodes_from=[custom_node])
    node_final = SecondaryNode('ridge', nodes_from=[lagged_node])
    pipeline = Pipeline(node_final)
    return pipeline


def get_input_data():
    test_file_path = str(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(test_file_path, '../../data/simple_sea_level.csv'))
    time_series = np.array(df['Level'])
    len_forecast = 50
    train_data = time_series[:-len_forecast]
    train_input, predict_input, _ = prepare_input_data(len_forecast=len_forecast,
                                                       train_data_features=train_data,
                                                       train_data_target=train_data,
                                                       test_data_features=train_data)
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
