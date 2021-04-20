from typing import Optional

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation
from fedot.core.log import Log, default_log


class LaggedTransformationImplementation(DataOperationImplementation):
    """ Implementation of lagged transformation for time series forecasting"""

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__()

        if not params:
            # Default parameters
            self.window_size = 10
        else:
            self.window_size = int(round(params.get('window_size')))

        # Define logger object
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for transformation of time series to lagged form

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with transformed features table
        """
        parameters = input_data.task.task_params
        old_idx = input_data.idx
        forecast_length = parameters.forecast_length

        # Correct window size parameter
        self.check_and_correct_window_size(input_data, forecast_length)

        if is_fit_chain_stage:
            # Transformation for fit stage of the chain
            target = input_data.target
            features = np.array(input_data.features)
            # Prepare features for training
            new_idx, features_columns = _ts_to_table(idx=old_idx,
                                                     time_series=features,
                                                     window_size=self.window_size)

            # Transform target
            new_idx, features_columns, new_target = _prepare_target(idx=new_idx,
                                                                    features_columns=features_columns,
                                                                    target=target,
                                                                    forecast_length=forecast_length)
            # Update target for Input Data
            input_data.target = new_target
            input_data.idx = new_idx
        else:
            # Transformation for predict stage of the chain
            features = np.array(input_data.features)
            features_columns = features[-self.window_size:]
            features_columns = features_columns.reshape(1, -1)

        output_data = self._convert_to_output(input_data,
                                              features_columns,
                                              data_type=DataTypesEnum.table)
        return output_data

    def check_and_correct_window_size(self, input_data, forecast_length):
        """ Method check if the length of the time series is not enough for
        lagged transformation - clip it

        :param input_data: InputData for transformation
        :param forecast_length: forecast length
        """
        removing_len = self.window_size + forecast_length
        if removing_len > len(input_data.features):
            previous_size = self.window_size
            # At least 10 objects we need for training, so minus 10
            self.window_size = len(input_data.features) - forecast_length - 10

            prefix = "Warning: window size of lagged transformation was changed"
            self.log.info(f"{prefix} from {previous_size} to {self.window_size}")

    def get_params(self):
        return {'window_size': self.window_size}


class TsSmoothingImplementation(DataOperationImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()

        if not params:
            # Default parameters
            self.window_size = 10
        else:
            self.window_size = int(round(params.get('window_size')))

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with smoothed time series
        """

        source_ts = pd.Series(input_data.features)

        # Apply smoothing operation
        smoothed_ts = source_ts.rolling(window=self.window_size).mean()
        smoothed_ts = np.array(smoothed_ts)

        # Filling first nans with source values
        smoothed_ts[:self.window_size] = source_ts[:self.window_size]

        output_data = self._convert_to_output(input_data,
                                              np.ravel(smoothed_ts),
                                              data_type=DataTypesEnum.ts)

        return output_data

    def get_params(self):
        return {'window_size': self.window_size}


class ExogDataTransformationImplementation(DataOperationImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for representing time series as column

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with features as columns
        """
        parameters = input_data.task.task_params
        old_idx = input_data.idx
        forecast_length = parameters.forecast_length

        if is_fit_chain_stage is True:
            # Transform features in "target-like way"
            _, _, features_columns = _prepare_target(idx=old_idx,
                                                     features_columns=input_data.features,
                                                     target=input_data.features,
                                                     forecast_length=forecast_length)

            # Transform target
            new_idx, _, new_target = _prepare_target(idx=old_idx,
                                                     features_columns=input_data.features,
                                                     target=input_data.target,
                                                     forecast_length=forecast_length)
            # Update target for Input Data
            input_data.target = new_target
            input_data.idx = new_idx
        else:
            # Transformation for predict stage of the chain
            features_columns = np.array(input_data.features)
            features_columns = features_columns.reshape(1, -1)

        output_data = self._convert_to_output(input_data,
                                              features_columns,
                                              data_type=DataTypesEnum.table)

        return output_data

    def get_params(self):
        return None


class GaussianFilterImplementation(DataOperationImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()

        if not params:
            # Default parameters
            self.sigma = 1
        else:
            self.sigma = int(round(params.get('sigma')))

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with smoothed time series
        """

        source_ts = np.array(input_data.features)

        # Apply smoothing operation
        smoothed_ts = gaussian_filter(source_ts, sigma=self.sigma)
        smoothed_ts = np.array(smoothed_ts)

        output_data = self._convert_to_output(input_data,
                                              np.ravel(smoothed_ts),
                                              data_type=DataTypesEnum.ts)

        return output_data

    def get_params(self):
        return {'sigma': self.sigma}


def _ts_to_table(idx, time_series, window_size):
    """ Method convert time series to lagged form.

    :param idx: the indices of the time series to convert
    :param time_series: source time series
    :param window_size: size of sliding window, which defines lag

    :return updated_idx: clipped indices of time series
    :return features_columns: lagged time series feature table
    """

    # Convert data to lagged form
    lagged_dataframe = pd.DataFrame({'t_id': time_series})
    vals = lagged_dataframe['t_id']
    for i in range(1, window_size + 1):
        frames = [lagged_dataframe, vals.shift(i)]
        lagged_dataframe = pd.concat(frames, axis=1)

    # Remove incomplete rows
    lagged_dataframe.dropna(inplace=True)

    transformed = np.array(lagged_dataframe)

    # Generate dataset with features
    features_columns = transformed[:, 1:]
    features_columns = np.fliplr(features_columns)

    # First n elements in time series are removed
    updated_idx = idx[window_size:]

    return updated_idx, features_columns


def _prepare_target(idx, features_columns, target, forecast_length):
    """ Method convert time series to lagged form. Transformation applied
    only for generating target table (time series considering as multi-target
    regression task).

    :param idx: remaining indices after lagged feature table generation
    :param features_columns: lagged feature table
    :param target: source time series
    :param forecast_length: forecast length

    :return updated_idx: clipped indices of time series
    :return updated_features: clipped lagged feature table
    :return updated_target: lagged target table
    """

    # Update target (clip first "window size" values)
    ts_target = target[idx]

    # Multi-target transformation
    if forecast_length > 1:
        # Target transformation
        df = pd.DataFrame({'t_id': ts_target})
        vals = df['t_id']
        for i in range(1, forecast_length):
            frames = [df, vals.shift(-i)]
            df = pd.concat(frames, axis=1)

        # Remove incomplete rows
        df.dropna(inplace=True)
        updated_target = np.array(df)

        threshold = -forecast_length + 1
        updated_idx = idx[: threshold]
        updated_features = features_columns[: threshold]
    else:
        updated_idx = idx
        updated_features = features_columns
        updated_target = ts_target

    return updated_idx, updated_features, updated_target
