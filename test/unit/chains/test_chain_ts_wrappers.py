import numpy as np
from sklearn.metrics import mean_absolute_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_ts_wrappers import out_of_sample_ts_forecast, in_sample_ts_forecast
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def prepare_input_data(forecast_length):
    ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 101])

    # Forecast for 2 elements ahead
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    train_input = InputData(idx=np.arange(0, len(ts)),
                            features=ts,
                            target=ts,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(ts)
    end_forecast = start_forecast + 2
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=ts,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input


def get_simple_short_lagged_chain():
    # Create simple chain for forecasting
    node_lagged = PrimaryNode('lagged')
    # Use 4 elements in time series as predictors
    node_lagged.custom_params = {'window_size': 4}
    node_final = SecondaryNode('linear', nodes_from=[node_lagged])
    chain = Chain(node_final)

    return chain


def test_out_of_sample_ts_forecast_correct():
    simple_length = 2
    multi_length = 10
    train_input, predict_input = prepare_input_data(simple_length)

    chain = get_simple_short_lagged_chain()
    chain.fit(train_input)

    # Make simple prediction
    simple_predict = chain.predict(predict_input)
    simple_predicted = np.ravel(np.array(simple_predict.predict))

    # Make multi-step forecast for 10 elements (2 * 5 steps)
    multi_predicted = out_of_sample_ts_forecast(chain=chain,
                                                input_data=predict_input,
                                                horizon=multi_length)

    assert len(simple_predicted) == simple_length
    assert len(multi_predicted) == multi_length


def test_in_sample_ts_forecast_correct():
    simple_length = 2
    multi_length = 10
    train_input, predict_input = prepare_input_data(simple_length)

    chain = get_simple_short_lagged_chain()
    chain.fit(train_input)

    multi_predicted = in_sample_ts_forecast(chain=chain,
                                            input_data=predict_input,
                                            horizon=multi_length)

    # Take validation part of time series
    time_series = np.array(train_input.features)
    validation_part = time_series[-multi_length:]

    metric = mean_absolute_error(validation_part, multi_predicted)
    is_forecast_correct = True

    assert is_forecast_correct
