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


def custom_model_imitation(train_data, test_data, params):
    a = params.get('a')
    b = params.get('b')
    shape = train_data.shape
    res = np.random.rand(shape[0], shape[1])*a + b
    return res

def get_custom_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge

    """
    lagged_node = PrimaryNode('lagged')
    custom_node = SecondaryNode('default', nodes_from=[lagged_node])
    custom_node.custom_params = {'params': {"a": 2, "b": 3}, 'model': custom_model_imitation}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

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
    predict_input = InputData(idx=np.arange(0, end_forecast),
                              features=np.concatenate([train_data_features, test_data_features]),
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def run_model():
    df = pd.read_csv('cases/data/time_series/metocean.csv')
    time_series = np.array(df['value'])
    len_forecast = 50
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)
    pipeline = get_custom_pipeline()
    pipeline.fit_from_scratch(train_input)
    pipeline.print_structure()
    predicted_values = pipeline.predict(predict_input)
    print(predicted_values.predict)

    pipeline_tuner = PipelineTuner(pipeline=pipeline, task=task,
                                   iterations=10)
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                            loss_function=mean_squared_error,
                                            loss_params={'squared': False},
                                            cv_folds=3,
                                            validation_blocks=3)

    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict
    predicted_values = pipeline.predict(predict_input)
    pipeline.print_structure()
    print(predicted_values.predict)


if __name__ == '__main__':
    run_model()