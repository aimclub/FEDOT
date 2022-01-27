import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import fitted_values
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, out_of_sample_ts_forecast, in_sample_fitted_values
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def prepare_ts_for_in_sample(forecast_length: int, horizon: int):
    """ Prepare input data for time series forecasting task. In-sample forecasting
    can be applied on this data

    :param forecast_length: forecast horizon for model
    :param horizon: horizon for validation
    """
    ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 101])

    # Forecast for 2 elements ahead
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # To avoid data leak
    ts_train = ts[:-horizon]
    train_input = InputData(idx=np.arange(0, len(ts_train)), features=ts_train,
                            target=ts_train, task=task, data_type=DataTypesEnum.ts)

    start_forecast = len(ts_train)
    end_forecast = start_forecast + forecast_length
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=ts, target=None, task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input


def get_simple_short_lagged_pipeline():
    # Create simple pipeline for forecasting
    node_lagged = PrimaryNode('lagged')
    # Use 4 elements in time series as predictors
    node_lagged.custom_params = {'window_size': 4}
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)

    return pipeline


def get_ts_pipelines_for_testing():
    """ Generate simple specific pipelines for testing """
    node_lagged = PrimaryNode('sparse_lagged')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    sparse_lagged_pipeline = Pipeline(node_final)

    node_lagged = PrimaryNode('lagged')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    lagged_pipeline = Pipeline(node_final)

    arima_pipeline = Pipeline(PrimaryNode('arima'))
    return arima_pipeline, sparse_lagged_pipeline, lagged_pipeline


def test_out_of_sample_ts_forecast_correct():
    simple_length = 2
    multi_length = 10
    train_input, predict_input = prepare_ts_for_in_sample(simple_length, multi_length)

    pipeline = get_simple_short_lagged_pipeline()
    pipeline.fit(train_input)

    # Make simple prediction
    simple_predict = pipeline.predict(predict_input)
    simple_predicted = np.ravel(np.array(simple_predict.predict))

    # Make multi-step forecast for 10 elements (2 * 5 steps)
    multi_predicted = out_of_sample_ts_forecast(pipeline=pipeline,
                                                input_data=predict_input,
                                                horizon=multi_length)

    assert len(simple_predicted) == simple_length
    assert len(multi_predicted) == multi_length


def test_in_sample_ts_forecast_correct():
    simple_length = 2
    multi_length = 10
    train_input, predict_input = prepare_ts_for_in_sample(simple_length, multi_length)

    pipeline = get_simple_short_lagged_pipeline()
    pipeline.fit(train_input)

    multi_predicted = in_sample_ts_forecast(pipeline=pipeline,
                                            input_data=predict_input,
                                            horizon=multi_length)

    # Take validation part of time series
    time_series = np.array(train_input.features)
    validation_part = time_series[-multi_length:]

    metric = mean_absolute_error(validation_part, multi_predicted)
    is_forecast_correct = True

    assert is_forecast_correct


def test_in_sample_ts_models_forecast_correct():
    """
    To be applied in in-sample forecasting method for AR, ARIMA and some other
    models it is required to use refit after each prediction
    """
    simple_length = 2
    multi_length = 10
    train_input, predict_input = prepare_ts_for_in_sample(simple_length, multi_length)

    ts_pipelines = get_ts_pipelines_for_testing()
    full_ts = predict_input.features
    for ts_pipeline in ts_pipelines:
        ts_pipeline.fit(train_input)

        multi_predicted = in_sample_ts_forecast(pipeline=ts_pipeline,
                                                input_data=predict_input,
                                                horizon=multi_length)

        # Validate without last element
        metric = mean_absolute_percentage_error(full_ts[-multi_length: -1], multi_predicted[: -1])
        # Metric should be low - forecast should almost repeat the actual line
        assert metric < 0.001


def test_fitted_values_correct():
    simple_length, multi_length = 5, 5
    ts_input, _ = prepare_ts_for_in_sample(simple_length, multi_length)

    for fitted_func in [fitted_values, in_sample_fitted_values]:
        pipeline = get_simple_short_lagged_pipeline()
        train_predicted = pipeline.fit(ts_input)

        fitted_ts_values = fitted_func(ts_input, train_predicted)

        assert len(fitted_ts_values.predict) == len(ts_input.target) - 4
