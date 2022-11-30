import math
from copy import copy
from typing import Union, Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


def out_of_sample_ts_forecast(pipeline: Pipeline, input_data: Union[InputData, MultiModalData],
                              horizon: int = None) -> np.array:
    """
    Method allow make forecast with appropriate forecast length. The previously
    predicted parts of the time series are used for forecasting next parts. Available
    only for time series forecasting task. Steps ahead provided iteratively.
    time series ----------------|
    forecast                    |---|---|---|

    :param pipeline: Pipeline for making time series forecasting
    :param input_data: data for prediction
    :param horizon: forecasting horizon
    :return final_forecast: array with forecast
    """
    task = input_data.task
    exception_if_not_ts_task(task)

    # How many elements to the future pipeline can produce
    forecast_length = task.task_params.forecast_length

    final_forecast = []
    if isinstance(input_data, MultiModalData) or input_data.data_type == DataTypesEnum.multi_ts:
        if forecast_length < horizon:
            raise ValueError('In case of multi-modal time-series and multi time-series'
                             'forecast horizon can not be bigger than forecast length model was fitted for.\n'
                             f'forecast_length = {forecast_length}\n'
                             f'horizon = {horizon}')
        iter_predict = pipeline.predict(input_data).predict
        iter_predict = np.ravel(np.array(iter_predict))
        final_forecast.append(iter_predict)
    else:
        pre_history_ts = np.array(input_data.features)
        number_of_iterations = math.ceil(horizon / forecast_length)
        # Make forecast iteratively moving throw the horizon
        for _ in range(0, number_of_iterations):
            iter_predict = pipeline.predict(input_data=input_data).predict
            iter_predict = np.ravel(np.array(iter_predict))
            final_forecast.append(iter_predict)

            # Add prediction to the historical data - update it
            pre_history_ts = np.hstack((pre_history_ts, iter_predict))

            # Prepare InputData for next iteration
            input_data = _update_input(pre_history_ts, forecast_length, task)

    # Create output data
    final_forecast = np.ravel(np.array(final_forecast))
    if final_forecast.ndim > 1:
        final_forecast = np.squeeze(final_forecast)
    # Clip the forecast if it is necessary
    final_forecast = final_forecast[:horizon]
    return final_forecast


def in_sample_ts_forecast(pipeline, input_data: Union[InputData, MultiModalData],
                          horizon: int = None) -> np.array:
    """
    Method allows to make in-sample forecasting. The actual values of the time
    series, rather than the previously predicted parts of the time series,
    are used for forecasting next parts. Sources indices in input_data parameter
    will be ignored
    time series ----------------|---|---|---|
    forecast                    |---|---|---|

    :param pipeline: Pipeline for making time series forecasting
    :param input_data: data for prediction
    :param horizon: forecasting horizon
    :return final_forecast: array with forecast
    """
    # Divide data on samples into pre-history and validation part
    task = input_data.task
    exception_if_not_ts_task(task)
    # How many elements to the future pipeline can produce
    scope_len = task.task_params.forecast_length
    number_of_iterations = _calculate_number_of_steps(scope_len, horizon)

    if isinstance(input_data, InputData):
        time_series = np.array(input_data.features)
        pre_history_ts = time_series[:-horizon]
        source_len = len(pre_history_ts)
        last_index_pre_history = source_len - 1

        data = _update_input(pre_history_ts, scope_len, task, input_data.data_type)
    else:
        # TODO simplify

        data = MultiModalData()
        for data_id in input_data.keys():
            features = input_data[data_id].features
            time_series = np.array(features)
            pre_history_ts = time_series[:-horizon]
            source_len = len(pre_history_ts)
            last_index_pre_history = source_len - 1

            local_data = _update_input(pre_history_ts, scope_len, task, input_data[data_id].data_type)
            data[data_id] = local_data

    # Calculate intervals
    intervals = _calculate_intervals(last_index_pre_history,
                                     number_of_iterations,
                                     scope_len)

    # Make forecast iteratively moving throw the horizon
    final_forecast = []
    for _, border in zip(range(0, number_of_iterations), intervals):

        iter_predict = pipeline.predict(input_data=data)
        iter_predict = np.ravel(np.array(iter_predict.predict))
        final_forecast.append(iter_predict)

        if isinstance(input_data, InputData):
            # Add actual values to the historical data - update it
            pre_history_ts = time_series[:border + 1]
            # Prepare InputData for next iteration
            data = _update_input(pre_history_ts, scope_len, task, input_data.data_type)
        else:
            # TODO simplify
            data = MultiModalData()
            for data_id in input_data.keys():
                features = input_data[data_id].features
                time_series = np.array(features)
                pre_history_ts = time_series[:border + 1]
                local_data = _update_input(pre_history_ts, scope_len, task, input_data[data_id].data_type)
                data[data_id] = local_data

    # Create output data
    final_forecast = np.ravel(np.array(final_forecast))
    # Clip the forecast if it is necessary
    final_forecast = final_forecast[:horizon]
    return final_forecast


def fitted_values(source_input: InputData, train_predicted: OutputData, horizon_step: int = None) -> OutputData:
    """ The method converts a multidimensional lagged array into an
    one-dimensional array - time series based on predicted values for training sample

    :param source_input: InputData which were used to train pipeline
    :param train_predicted: OutputData with trained values
    :param horizon_step: index of elements for forecast. If None - perform
    averaging for all forecasting steps
    """
    copied_data = copy(train_predicted)
    if horizon_step is not None:
        # Take particular forecast step
        copied_data.predict = copied_data.predict[:, horizon_step]
        if isinstance(copied_data.idx, list):
            # if indices can not be incremented, replace it
            copied_data.idx = generate_ids(source_input, copied_data, expand=False)
        copied_data.idx = copied_data.idx + horizon_step
        return copied_data
    else:
        # Perform collapse with averaging
        forecast_length = copied_data.task.task_params.forecast_length

        # Extend source index range
        if isinstance(copied_data.idx, list):
            indices_range = generate_ids(source_input, copied_data, expand=True)
        else:
            # if indices can be incremented
            indices_range = np.arange(copied_data.idx[0],
                                      copied_data.idx[-1] + forecast_length)

        # Lagged matrix with indices in cells
        _, idx_matrix = ts_to_table(idx=indices_range,
                                    time_series=indices_range,
                                    window_size=forecast_length)
        predicted_matrix = copied_data.predict

        # For every index calculate mean predictions (by all forecast steps)
        final_predictions = []
        for index in indices_range:
            vals = predicted_matrix[idx_matrix == index]
            mean_value = np.mean(vals)
            final_predictions.append(mean_value)
        copied_data.predict = np.array(final_predictions)
        copied_data.idx = indices_range
        return copied_data


def in_sample_fitted_values(source_input: InputData, train_predicted: OutputData) -> OutputData:
    """ Perform in sample validation based on training sample """
    forecast_length = train_predicted.task.task_params.forecast_length
    all_values = []
    step = 0
    # Glues together parts of predictions using "in-sample" way
    while step < len(train_predicted.predict):
        all_values.extend(train_predicted.predict[step, :])
        step += forecast_length

    # In some cases it doesn't reach the end
    if not np.isclose(all_values[-1], train_predicted.predict[-1, -1]):
        missing_part_index = step - len(train_predicted.predict) + 1
        # Store missing predicted values
        all_values.extend(train_predicted.predict[-1, missing_part_index:])

    copied_data = copy(train_predicted)
    copied_data.predict = np.array(all_values)
    # Update indices
    first_id = copied_data.idx[0]
    if isinstance(first_id, str):
        indices_range = generate_ids(source_input, copied_data, expand=True)
        copied_data.idx = indices_range[:-1]
    else:
        copied_data.idx = np.arange(first_id, first_id + len(all_values))

    return copied_data


def _calculate_number_of_steps(scope_len, horizon):
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


def _update_input(pre_history_ts, scope_len, task, data_type: DataTypesEnum = DataTypesEnum.ts):
    """ Method make new InputData object based on the previous part of time
    series

    Args:
        pre_history_ts: time series
        scope_len: how many elements to the future can algorithm forecast
        task: time series forecasting task
        data_type: type of data (ts or multi_ts

    Returns:
        input_data: updated InputData
    """
    start_forecast = len(pre_history_ts)
    end_forecast = start_forecast + scope_len
    input_data = InputData(idx=np.arange(start_forecast, end_forecast),
                           features=pre_history_ts, target=None,
                           task=task, data_type=data_type)

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
    if task.task_type is not TaskTypesEnum.ts_forecasting:
        raise ValueError('Method forecast is available only for time series forecasting task')


def generate_ids(source_input, output_data, expand: bool):
    """
    Create new indices for fitted time series values. Process shift after
    starting clipping.

    :param source_input: source idx, features and target
    :param output_data: OutputData after pipeline fit
    :param expand: there is a need to take into account the forecasting horizon.
    If True, expand indices.
    """
    forecast_len = source_input.task.task_params.forecast_length
    source_idx = source_input.idx[:-forecast_len]

    # Calculate difference between source idx len and new
    clipped_starting_values = len(source_idx) - len(output_data.idx)

    if expand:
        indices_range = np.arange(clipped_starting_values + 1,
                                  len(source_idx) + forecast_len + 1)
    else:
        indices_range = np.arange(clipped_starting_values + 1,
                                  len(source_idx) + 1)
    return indices_range


def convert_forecast_to_output(pre_history_data: Union[InputData, MultiModalData], forecast: np.array,
                               idx: Optional[np.array] = None) -> OutputData:
    """Converts forecast array to OutputData

    Args:
        pre_history_data: data which was used for prediction
        forecast: array with predicted values
        idx: array with idx values. If None sets next idx after `pre_history_data` with length of forecast for InputData
        and idx from 0 to forecast length for MultimodalData.
    """
    features = pre_history_data.features if isinstance(pre_history_data, InputData) else None
    if forecast.ndim > 1:
        forecast = np.squeeze(forecast)
    if idx is None:
        if features is not None:
            idx = np.arange(len(features), len(features) + len(forecast))
        else:
            idx = np.arange(len(forecast))
    prediction = OutputData(idx=idx,
                            features=features,
                            predict=forecast,
                            task=pre_history_data.task,
                            data_type=DataTypesEnum.ts)
    return prediction
