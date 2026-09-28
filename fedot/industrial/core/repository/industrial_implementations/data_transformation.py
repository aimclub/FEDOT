from copy import copy

import pandas as pd
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix


# Method for lagged transformation
def transform_lagged_for_fit_industrial(self, input_data: InputData) -> OutputData:
    """Method for transformation of time series to lagged form for fit stage

    Args:
        input_data: data with features, target and ids to process

    Returns:
        output data with transformed features table
    """
    train_data = copy(input_data)
    forecast_length = train_data.task.task_params.forecast_length
    # Correct window size parameter
    if self.window_size == 0:
        self._check_and_correct_window_size(
            train_data.features, forecast_length)
    else:
        self.log.info(("Window size dont change"))
    lagged_features = HankelMatrix(
        time_series=train_data.features,
        window_size=self.window_size).trajectory_matrix.T
    lagged_target = HankelMatrix(
        time_series=train_data.features[self.window_size:],
        window_size=forecast_length).trajectory_matrix.T
    lagged_features = lagged_features[:lagged_target.shape[0], :]
    train_data.target = lagged_target
    output_data = self._convert_to_output(train_data,
                                          lagged_features,
                                          data_type=DataTypesEnum.table)
    return output_data


def transform_lagged_industrial(self, input_data: InputData):
    train_data = copy(input_data)
    # forecast_length = train_data.task.task_params.forecast_length
    lagged_features = HankelMatrix(
        time_series=train_data.features,
        window_size=self.window_size).trajectory_matrix.T
    if input_data.target is not None:
        if len(input_data.target.shape) < 2:
            lagged_target = HankelMatrix(
                time_series=train_data.features[self.window_size:],
                window_size=train_data.task.task_params.forecast_length).trajectory_matrix.T
            lagged_features = lagged_features[:lagged_target.shape[0], :]
            train_data.target = lagged_target
    output_data = self._convert_to_output(train_data,
                                          lagged_features,
                                          data_type=DataTypesEnum.table)
    return output_data


def _check_and_correct_window_size_industrial(
        self,
        time_series: np.ndarray,
        forecast_length: int):
    """ Method check if the length of the time series is not enough for
        lagged transformation

        Args:
            time_series: time series for transformation
            forecast_length: forecast length

        Returns:

        """
    max_ws = round(len(time_series) / 2)  # half of all ts
    # 5 percent of all ts or 2 elements
    min_ws = max(round(len(time_series) * 0.05), 2)
    max_allowed_window_size = max(min_ws, max_ws)
    step = round(1.5 * forecast_length)
    range_ws = max_allowed_window_size - min_ws
    if step > range_ws:
        step = round(range_ws * 0.5)
    window_list = list(range(min_ws, max_allowed_window_size, step))

    if self.window_size == 0 or self.window_size > max_allowed_window_size:
        try:
            window_size = np.random.choice(window_list)
        except Exception:
            window_size = 3 * forecast_length
        window_size = max(window_size, 4)
        self.log.message(
            (f"Window size of lagged transformation was changed "
             f"by WindowSizeSelector from {self.params.get('window_size')} to {window_size}"))
        self.params.update(window_size=window_size)

    # Minimum threshold
    if self.window_size < self.window_size_minimum:
        self.log.info(
            (f"Warning: window size of lagged transformation was changed "
             f"from {self.params.get('window_size')} to {self.window_size_minimum}"))
        self.params.update(window_size=self.window_size_minimum)


def transform_smoothing_industrial(self, input_data: InputData) -> OutputData:
    """Method for smoothing time series

    Args:
        input_data: data with features, target and ids to process

    Returns:
        output data with smoothed time series
    """

    source_ts = input_data.features
    if input_data.data_type == DataTypesEnum.multi_ts:
        full_smoothed_ts = []
        for ts_n in range(source_ts.shape[1]):
            ts = pd.Series(source_ts[:, ts_n])
            smoothed_ts = self._apply_smoothing_to_series(ts)
            full_smoothed_ts.append(smoothed_ts)
        output_data = self._convert_to_output(input_data,
                                              np.array(full_smoothed_ts).T,
                                              data_type=input_data.data_type)
    else:
        source_ts = pd.Series(input_data.features.flatten())
        smoothed_ts = np.ravel(self._apply_smoothing_to_series(source_ts))
        smoothed_ts = smoothed_ts.reshape(1, -1)
        output_data = self._convert_to_output(input_data,
                                              smoothed_ts,
                                              data_type=input_data.data_type)

    return output_data
