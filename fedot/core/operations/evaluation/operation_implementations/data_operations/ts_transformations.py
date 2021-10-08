from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import TruncatedSVD

from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData


class LaggedImplementation(DataOperationImplementation):

    def __init__(self, log: Log = None, **params):
        super().__init__()

        self.window_size_minimum = None
        self.window_size = None
        self.n_components = None
        self.sparse_transform = False
        self.use_svd = False
        self.features_columns = None
        self.parameters_changed = False

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

    def transform(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for transformation of time series to lagged form

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with transformed features table
        """

        new_input_data = copy(input_data)
        forecast_length = new_input_data.task.task_params.forecast_length
        old_idx = new_input_data.idx

        # Correct window size parameter
        self.window_size, self.parameters_changed = _check_and_correct_window_size(new_input_data.features,
                                                                                   self.window_size,
                                                                                   forecast_length,
                                                                                   self.window_size_minimum,
                                                                                   self.log)

        if is_fit_pipeline_stage:
            # Transformation for fit stage of the pipeline
            target = new_input_data.target
            features = np.array(new_input_data.features)
            # Prepare features for training
            new_idx, self.features_columns = ts_to_table(idx=old_idx,
                                                         time_series=features,
                                                         window_size=self.window_size)

            # Sparsing matrix of lagged features
            if self.sparse_transform:
                self.features_columns = _sparse_matrix(self.log, self.features_columns, self.n_components, self.use_svd)
            # Transform target
            new_idx, self.features_columns, new_target = prepare_target(idx=new_idx,
                                                                        features_columns=self.features_columns,
                                                                        target=target,
                                                                        forecast_length=forecast_length)

            # Update target for Input Data
            new_input_data.target = new_target
            new_input_data.idx = new_idx
        else:
            # Transformation for predict stage of the pipeline
            if self.sparse_transform:
                self.features_columns = self.features_columns[-1]

            if not self.sparse_transform:
                features = np.array(new_input_data.features)
                self.features_columns = features[-self.window_size:]

            self.features_columns = self.features_columns.reshape(1, -1)

        output_data = self._convert_to_output(new_input_data,
                                              self.features_columns,
                                              data_type=DataTypesEnum.table)
        return output_data


class SparseLaggedTransformationImplementation(LaggedImplementation):
    """ Implementation of sparse lagged transformation for time series forecasting"""

    def __init__(self, **params):
        super().__init__()
        self.sparse_transform = True
        self.window_size_minimum = 6

        self.window_size = round(params.get('window_size'))
        self.n_components = params.get('n_components')

    def get_params(self):
        params_dict = {'window_size': self.window_size,
                       'n_components': self.n_components,
                       'use_svd': self.use_svd}
        if self.parameters_changed is True:
            return tuple([params_dict, ['window_size']])
        else:
            return params_dict


class LaggedTransformationImplementation(LaggedImplementation):
    """ Implementation of lagged transformation for time series forecasting"""

    def __init__(self, **params):
        super().__init__()
        self.window_size_minimum = 2
        self.window_size = round(params.get('window_size'))

    def get_params(self):
        params_dict = {'window_size': self.window_size}
        if self.parameters_changed is True:
            return tuple([params_dict, ['window_size']])
        else:
            return params_dict


class TsSmoothingImplementation(DataOperationImplementation):

    def __init__(self, **params):
        super().__init__()

        if not params:
            # Default parameters
            self.window_size = 10
        else:
            self.window_size = round(params.get('window_size'))

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData, is_fit_pipeline_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
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

    def __init__(self, **params):
        super().__init__()

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for representing time series as column

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with features as columns
        """
        copied_data = copy(input_data)
        parameters = copied_data.task.task_params
        old_idx = copied_data.idx
        forecast_length = parameters.forecast_length

        if is_fit_pipeline_stage is True:
            # Transform features in "target-like way"
            _, _, features_columns = prepare_target(idx=old_idx,
                                                    features_columns=copied_data.features,
                                                    target=copied_data.features,
                                                    forecast_length=forecast_length)

            # Transform target
            new_idx, _, new_target = prepare_target(idx=old_idx,
                                                    features_columns=copied_data.features,
                                                    target=copied_data.target,
                                                    forecast_length=forecast_length)
            # Update target for Input Data
            copied_data.target = new_target
            copied_data.idx = new_idx
        else:
            # Transformation for predict stage of the pipeline
            features_columns = np.array(copied_data.features)[-forecast_length:]
            copied_data.idx = copied_data.idx[-forecast_length:]
            features_columns = features_columns.reshape(1, -1)

        output_data = self._convert_to_output(copied_data,
                                              features_columns,
                                              data_type=DataTypesEnum.table)

        return output_data

    def get_params(self):
        return {}


class GaussianFilterImplementation(DataOperationImplementation):

    def __init__(self, **params):
        super().__init__()

        if not params:
            # Default parameters
            self.sigma = 1
        else:
            self.sigma = round(params.get('sigma'))

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData, is_fit_pipeline_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
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


def _check_and_correct_window_size(time_series: np.array, window_size: int, forecast_length: int,
                                   window_size_minimum: int, log: Log):
    """ Method check if the length of the time series is not enough for
    lagged transformation - clip it

    :param time_series: time series for transformation
    :param window_size: size of sliding window, which defines lag
    :param forecast_length: forecast length
    :param window_size_minimum: minimum moving window size
    :param log: logger for saving messages
    """
    prefix = "Warning: window size of lagged transformation was changed"
    was_changed = False

    # Maximum threshold
    removing_len = window_size + forecast_length
    if removing_len > len(time_series):
        previous_size = window_size
        # At least 10 objects we need for training, so minus 10
        window_size = len(time_series) - forecast_length - 10

        log.info(f"{prefix} from {previous_size} to {window_size}")
        was_changed = True

    # Minimum threshold
    if window_size < window_size_minimum:
        previous_size = window_size
        window_size = window_size_minimum

        log.info(f"{prefix} from {previous_size} to {window_size}")
        was_changed = True

    return window_size, was_changed


def ts_to_table(idx, time_series: np.array, window_size: int):
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

    # Remove extra column (it will go to target)
    features_columns = transformed[:, 1:]
    # Generate dataset with features
    features_columns = np.fliplr(features_columns)

    # First n elements in time series are removed
    updated_idx = idx[window_size:]

    return updated_idx, features_columns


def _sparse_matrix(logger, features_columns: np.array, n_components_perc=0.5, use_svd=False):
    """ Method converts the matrix to sparse form

        :param features_columns: matrix to sparse
        :param n_components_perc: initial approximation of percent of components to keep
        :param use_svd: is there need to use SVD method for sparse or use naive method

        :return components: reduced dimension matrix, its shape depends on the number of components which includes
                            the threshold of explained variance gain
        """
    if not n_components_perc:
        n_components_perc = 0.5

    if use_svd:
        n_components = int(features_columns.shape[1] * n_components_perc)
        # Getting the initial approximation of number of components
        if not n_components:
            n_components = int(features_columns.shape[1] * n_components_perc)
        if n_components >= features_columns.shape[0]:
            n_components = features_columns.shape[0] - 1
        logger.info(f'Initial approximation of number of components set as {n_components}')

        # Forming the first value of explained variance
        components = _get_svd(features_columns, n_components)
    else:
        step = int(1 / n_components_perc)
        indeces_to_stay = np.arange(1, features_columns.shape[1], step)
        components = np.take(features_columns, indeces_to_stay, 1)

    return components


def _get_svd(features_columns: np.array, n_components: int):
    """ Method converts the matrix to svd sparse form

    :param features_columns: matrix to sparse
    :param n_components: number of components to keep

    :return components: transformed sparse matrix
    """

    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
    svd.fit(features_columns.T)
    components = svd.components_.T
    return components


def prepare_target(idx, features_columns: np.array, target, forecast_length: int):
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
