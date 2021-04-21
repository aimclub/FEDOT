import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain import Chain


def out_of_sample_ts_forecast(chain: Chain, input_data: InputData,
                              horizon: int = None) -> np.array:
    """
    Method allow make forecast with appropriate forecast length. The previously
    predicted parts of the time series are used for forecasting next parts. Available
    only for time series forecasting task. Steps ahead provided iteratively.
    time series ----------------|
    forecast                    |---|---|---|

    :param chain: Chain for making time series forecasting
    :param input_data: data for prediction
    :param horizon: forecasting horizon
    :return final_forecast: array with forecast
    """
    # Prepare data for time series forecasting
    task = input_data.task
    exception_if_not_ts_task(task)

    pre_history_ts = np.array(input_data.features)
    source_len = len(pre_history_ts)

    # How many elements to the future chain can produce
    scope_len = task.task_params.forecast_length
    amount_of_iterations = _calculate_amount_of_steps(scope_len, horizon)

    # Make forecast iteratively moving throw the horizon
    final_forecast = []
    for _ in range(0, amount_of_iterations):
        iter_predict = chain.root_node.predict(input_data=input_data)
        iter_predict = np.ravel(np.array(iter_predict.predict))
        final_forecast.append(iter_predict)

        # Add prediction to the historical data - update it
        pre_history_ts = np.hstack((pre_history_ts, iter_predict))

        # Prepare InputData for next iteration
        input_data = _update_input(pre_history_ts, scope_len, task)

    # Create output data
    final_forecast = np.ravel(np.array(final_forecast))
    # Clip the forecast if it is necessary
    final_forecast = final_forecast[:horizon]
    return final_forecast


def in_sample_ts_forecast(chain: Chain, input_data: InputData,
                          horizon: int = None) -> np.array:
    """
    Method allows to make in-sample forecasting. The actual values of the time
    series, rather than the previously predicted parts of the time series,
    are used for forecasting next parts.
    time series ----------------|---|---|---|
    forecast                    |---|---|---|

    :param chain: Chain for making time series forecasting
    :param input_data: data for prediction
    :param horizon: forecasting horizon
    :return final_forecast: array with forecast
    """
    # Divide data on samples into pre-history and validation part
    task = input_data.task
    exception_if_not_ts_task(task)

    time_series = np.array(input_data.features)
    pre_history_ts = time_series[:-horizon]
    source_len = len(pre_history_ts)
    last_index_pre_history = source_len - 1

    # How many elements to the future chain can produce
    scope_len = task.task_params.forecast_length
    amount_of_iterations = _calculate_amount_of_steps(scope_len, horizon)

    # Calculate intervals
    intervals = _calculate_intervals(last_index_pre_history,
                                     amount_of_iterations,
                                     scope_len)

    data = _update_input(pre_history_ts, scope_len, task)
    # Make forecast iteratively moving throw the horizon
    final_forecast = []
    for _, border in zip(range(0, amount_of_iterations), intervals):
        iter_predict = chain.root_node.predict(input_data=data)
        iter_predict = np.ravel(np.array(iter_predict.predict))
        final_forecast.append(iter_predict)

        # Add prediction to the historical data - update it
        pre_history_ts = time_series[:border+1]

        # Prepare InputData for next iteration
        data = _update_input(pre_history_ts, scope_len, task)

    # Create output data
    final_forecast = np.ravel(np.array(final_forecast))
    # Clip the forecast if it is necessary
    final_forecast = final_forecast[:horizon]
    return final_forecast


def _calculate_amount_of_steps(scope_len, horizon):
    """ Method return amount of iterations which must be done for multistep
    time series forecasting

    :param scope_len: time series forecasting length
    :param horizon: forecast horizon
    :return amount_of_steps: amount of steps to produce
    """
    amount_of_iterations = int(horizon // scope_len)

    # Remainder of the division
    resid = int(horizon % scope_len)
    if resid == 0:
        amount_of_steps = amount_of_iterations
    else:
        amount_of_steps = amount_of_iterations + 1

    return amount_of_steps


def _update_input(pre_history_ts, scope_len, task):
    """ Method make new InputData object based on the previous part of time
    series

    :param pre_history_ts: time series
    :param scope_len: how many elements to the future can algorithm forecast
    :param task: time series forecasting task

    :return input_data: updated InputData
    """
    start_forecast = len(pre_history_ts)
    end_forecast = start_forecast + scope_len
    input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                           features=pre_history_ts, target=None,
                           task=task, data_type=DataTypesEnum.ts)

    return input_data


def _calculate_intervals(last_index_pre_history, amount_of_iterations, scope_len):
    """ Function calculate

    :param last_index_pre_history: last id of the known part of time series
    :param amount_of_iterations: amount of steps for time series forecasting
    :param scope_len: amount of elements in every time series forecasting step
    :return intervals: ids of finish of every step in time series
    """
    intervals = []
    current_border = last_index_pre_history
    for i in range(0, amount_of_iterations):
        current_border = current_border + scope_len
        intervals.append(current_border)

    return intervals


def exception_if_not_ts_task(task):
    if task.task_type != TaskTypesEnum.ts_forecasting:
        raise ValueError(f'Method forecast is available only for time series forecasting task')
