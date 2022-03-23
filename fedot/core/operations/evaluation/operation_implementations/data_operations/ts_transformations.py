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
from fedot.core.data.data import InputData, OutputData


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
            target = np.array(new_input_data.target)
            features = np.array(new_input_data.features)

            new_target, new_idx = self._apply_transformation_for_fit(new_input_data, features,
                                                                     target, forecast_length, old_idx)

            # Update target for Input Data
            new_input_data.target = new_target
            new_input_data.idx = new_idx
        else:
            # Transformation for predict stage of the pipeline
            self._apply_transformation_for_predict(new_input_data, forecast_length)

        output_data = self._convert_to_output(new_input_data,
                                              self.features_columns,
                                              data_type=DataTypesEnum.table)
        self._update_column_types(output_data)
        return output_data

    def _update_column_types(self, output_data: OutputData):
        """ Update column types after lagged transformation. All features becomes float """
        features_n_rows, features_n_cols = output_data.predict.shape
        features_column_types = [str(float)] * features_n_cols
        column_types = {'features': features_column_types}

        if output_data.target is not None and len(output_data.target.shape) > 1:
            target_n_rows, target_n_cols = output_data.target.shape
            column_types.update({'target': [str(float)] * target_n_cols})
        output_data.supplementary_data.column_types = column_types

    def _apply_transformation_for_fit(self, input_data: InputData, features: np.array, target: np.array,
                                      forecast_length: int, old_idx: np.array):
        """ Apply lagged transformation on each time series in the current dataset """
        # Shape of the time series
        if len(features.shape) > 1:
            # Multivariate time series
            n_elements, n_time_series = features.shape
        else:
            n_time_series = 1
        all_transformed_features = None
        for current_ts_id in range(n_time_series):
            # For each time series in the array
            if n_time_series == 1:
                current_ts = features
            else:
                current_ts = np.ravel(features[:, current_ts_id])

            # Prepare features for training
            new_idx, transformed_cols = ts_to_table(idx=old_idx,
                                                    time_series=current_ts,
                                                    window_size=self.window_size,
                                                    is_lag=True)

            # Sparsing matrix of lagged features
            if self.sparse_transform:
                transformed_cols = _sparse_matrix(self.log,
                                                  transformed_cols,
                                                  self.n_components,
                                                  self.use_svd)
            # Transform target
            current_target = self._current_target_for_each_ts(current_ts_id, target)
            new_idx, transformed_cols, new_target = prepare_target(all_idx=input_data.idx,
                                                                   idx=new_idx,
                                                                   features_columns=transformed_cols,
                                                                   target=current_target,
                                                                   forecast_length=forecast_length)
            if current_ts_id == 0:
                # Init full lagged table
                all_transformed_features = transformed_cols
                all_transformed_target = new_target
                all_transformed_idx = np.array(new_idx)
            else:
                all_transformed_features, all_transformed_target, all_transformed_idx = self.stack_by_type_fit(
                    input_data, all_transformed_features,
                    all_transformed_target, all_transformed_idx,
                    transformed_cols, new_target, new_idx)

        input_data.features = all_transformed_features
        self.features_columns = all_transformed_features
        return all_transformed_target, all_transformed_idx

    def stack_by_type_fit(self, input_data, all_features, all_target, all_idx, features, target, idx):
        """ Apply stack function for multi_ts and multivariable ts types on fit step"""
        functions_by_type = {
            DataTypesEnum.multi_ts: self._stack_multi_ts,
            DataTypesEnum.ts: self._stack_multi_variable
        }
        stack_function = functions_by_type.get(input_data.data_type)
        return stack_function(all_features, all_target, all_idx, features, target, idx)

    def _stack_multi_variable(self, all_features: np.array,
                              all_target: np.array,
                              all_idx: np.array,
                              features: np.array,
                              target: np.array,
                              idx: [list, np.array]):
        """
        Horizontally stack tables as multiple variables extends features for training

        :param all_features: array with all features for adding new
        :param all_target:  array with all target (does not change)
        :param all_idx: array with all indices (does not change)
        :param features: array with new features for adding
        :param target: array with new target for adding
        :param idx: array with new idx for adding
        """
        all_features = np.hstack((all_features, features))
        return all_features, all_target, all_idx

    def _stack_multi_ts(self, all_features: np.array,
                        all_target: np.array,
                        all_idx: np.array,
                        features: np.array,
                        target: np.array,
                        idx: [list, np.array]):
        """
        Vertically stack tables as multi_ts data extends training set as combination of train and target

        :param all_features: array with all features for adding new
        :param all_target:  array with all target
        :param all_idx: array with all indices
        :param features: array with new features for adding
        :param target: array with new target for adding
        :param idx: array with new idx for adding
        """
        all_features = np.vstack((all_features, features))
        all_target = np.vstack((all_target, target))
        all_idx = np.hstack((all_idx, np.array(idx)))
        return all_features, all_target, all_idx

    def _current_target_for_each_ts(self, current_ts_id, target):
        """ Returns target for each time-series"""
        if len(target.shape) > 1:
            # if multi_ts case
            if current_ts_id >= target.shape[1]:
                while current_ts_id >= target.shape[1]:
                    current_ts_id = current_ts_id - target.shape[1]
            return target[:, current_ts_id]
        else:
            # if multivariable case
            return target

    def _apply_transformation_for_predict(self, input_data: InputData, forecast_length: int):
        """ Apply lagged transformation for every column (time series) in the dataset """
        if self.sparse_transform:
            self.log.debug(f'Sparse lagged transformation applied. If new data were used. Call fit method')
            transformed_cols = self._update_features_for_sparse(input_data)
            # Take last row in the lagged table and reshape into array with 1 row and n columns
            self.features_columns = transformed_cols[-1].reshape(1, -1)
            return self.features_columns

        if len(input_data.features.shape) > 1:
            # Multivariate time series
            n_elements, n_time_series = input_data.features.shape
        else:
            n_time_series = 1

        all_transformed_features = None
        for current_ts_id in range(n_time_series):
            # For each time series
            if n_time_series == 1:
                current_ts = input_data.features
            else:
                current_ts = np.ravel(input_data.features[:, current_ts_id])

            # Take last window_size elements for current ts
            last_part_of_ts = current_ts[-self.window_size:].reshape(1, -1)
            if current_ts_id == 0:
                all_transformed_features = last_part_of_ts
            else:
                all_transformed_features = self.stack_by_type_predict(input_data,
                                                                      all_transformed_features,
                                                                      last_part_of_ts)

        if input_data.data_type == DataTypesEnum.multi_ts:
            all_transformed_features = np.expand_dims(all_transformed_features[0], axis=0)
        self.features_columns = all_transformed_features
        return all_transformed_features

    def stack_by_type_predict(self, input_data, all_features, part_to_add):
        """ Apply stack function for multi_ts and multivariable ts types on predict step"""
        if input_data.data_type == DataTypesEnum.multi_ts:
            # for mutli_ts
            all_features = np.vstack((all_features, part_to_add))
        if input_data.data_type == DataTypesEnum.ts:
            # for multivariable
            all_features = np.hstack((all_features, part_to_add))
        return all_features

    def _update_features_for_sparse(self, input_data: InputData):
        """ Make sparse matrix which will be used during forecasting """
        # Prepare features for training
        new_idx, transformed_cols = ts_to_table(idx=input_data.idx,
                                                time_series=input_data.features,
                                                window_size=self.window_size,
                                                is_lag=True)
        # Sparsing
        transformed_cols = _sparse_matrix(self.log,
                                          transformed_cols,
                                          self.n_components,
                                          self.use_svd)
        return transformed_cols


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
            source_ts = pd.Series(input_data.features)
            smoothed_ts = np.ravel(self._apply_smoothing_to_series(source_ts))
            output_data = self._convert_to_output(input_data,
                                                  smoothed_ts,
                                                  data_type=input_data.data_type)

        return output_data

    def _apply_smoothing_to_series(self, ts):
        smoothed_ts = ts.rolling(window=self.window_size).mean()
        smoothed_ts = np.array(smoothed_ts)

        # Filling first nans with source values
        smoothed_ts[:self.window_size] = ts[:self.window_size]
        return smoothed_ts

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
            _, _, features_columns = prepare_target(all_idx=input_data.idx,
                                                    idx=old_idx,
                                                    features_columns=copied_data.features,
                                                    target=copied_data.features,
                                                    forecast_length=forecast_length)

            # Transform target
            new_idx, _, new_target = prepare_target(all_idx=input_data.idx,
                                                    idx=old_idx,
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

        if input_data.data_type != DataTypesEnum.multi_ts:
            smoothed_ts = np.ravel(smoothed_ts)
        output_data = self._convert_to_output(input_data,
                                              smoothed_ts,
                                              data_type=input_data.data_type)

        return output_data

    def get_params(self):
        return {'sigma': self.sigma}


class NumericalDerivativeFilterImplementation(DataOperationImplementation):
    def __init__(self, log: Log = None, **params):
        super().__init__()
        self.params = params
        self.parameters_changed = False
        self.changed_params = []

        self.max_poly_degree = 5
        self.default_poly_degree = 2
        self.default_order = 1

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        self.poly_degree = int(self.params['poly_degree'])
        self.order = int(self.params['order'])
        self.window_size = int(self.params['window_size'])
        self._correct_params()

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData, is_fit_pipeline_stage: bool):
        """ Method for finding numerical derivative of time series

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """

        source_ts = np.array(input_data.features)
        # Apply differential operation
        if input_data.data_type == DataTypesEnum.multi_ts:
            full_differential_ts = []
            for ts_n in range(source_ts.shape[1]):
                ts = source_ts[:, ts_n]
                differential_ts = self._differential_filter(ts)
                full_differential_ts.append(differential_ts)
            output_data = self._convert_to_output(input_data,
                                                  np.array(full_differential_ts).T,
                                                  data_type=input_data.data_type)
        else:
            differential_ts = np.ravel(self._differential_filter(source_ts))
            output_data = self._convert_to_output(input_data,
                                                  differential_ts,
                                                  data_type=input_data.data_type)

        return output_data

    def _differential_filter(self, ts):
        """ NumericalDerivative filter """
        if self.window_size > ts.shape[0]:
            self.parameters_changed = True
            self.changed_params.append('window_size')
            self.log.info(f'NumericalDerivativeFilter: invalid parameter window_size ({self.window_size}) changed to '
                          f'{self.poly_degree + 1}')
            self.window_size = self.poly_degree + 1
        x = np.arange(ts.shape[0])

        ts_len = x.shape[0]
        der_f = np.zeros(ts_len)

        # Take the differentials in the center of the domain
        for center_window in range(self.window_size, ts_len - self.window_size):
            points = np.arange(center_window - self.window_size, center_window + self.window_size)
            # Fit to a Chebyshev polynomial
            # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
            poly = np.polynomial.chebyshev.Chebyshev.fit(x[points], ts[points], self.poly_degree,
                                                         window=[np.min(points), np.max(points)])
            der_f[center_window] = poly.deriv(m=self.order)(x[center_window])

        supp_1 = ts[0:self.window_size]
        coordsupp_1 = x[0:self.window_size]
        supp_2 = ts[-self.window_size:]
        coordsupp_2 = x[-self.window_size:]
        poly = np.polynomial.chebyshev.Chebyshev.fit(coordsupp_1, supp_1, self.window_size - 1)
        der_f[0:self.window_size] = poly.deriv(m=self.order)(coordsupp_1)
        poly = np.polynomial.chebyshev.Chebyshev.fit(coordsupp_2, supp_2, self.window_size - 1)
        der_f[-self.window_size:] = poly.deriv(m=self.order)(coordsupp_2)
        for _ in range(self.order):
            supp_1 = np.gradient(supp_1, coordsupp_1, edge_order=2)
            supp_2 = np.gradient(supp_2, coordsupp_2, edge_order=2)
        der_f[0:self.window_size] = supp_1
        der_f[-self.window_size:] = supp_2
        return np.transpose(der_f)

    def _correct_params(self):
        if self.poly_degree > 5:
            self.parameters_changed = True
            self.log.info(f'NumericalDerivativeFilter: invalid parameter poly_degree ({self.poly_degree}) '
                          f'changed to {self.max_poly_degree}')
            self.changed_params.append('poly_degree')
            self.poly_degree = self.max_poly_degree
        if self.order < 1:
            self.parameters_changed = True
            self.log.info(f'NumericalDerivativeFilter: invalid parameter order ({self.order}) '
                          f'changed to {self.default_order}')
            self.changed_params.append('order')
            self.order = self.default_order
        if self.order >= self.poly_degree:
            self.parameters_changed = True
            self.changed_params.append('order')
            self.log.info(f'NumericalDerivativeFilter: invalid parameter poly_degree ({self.poly_degree}) '
                          f'changed to {self.order + 1}')
            self.poly_degree = self.order + 1
        if self.window_size < self.poly_degree:
            self.parameters_changed = True
            self.changed_params.append('window_size')
            self.log.info(f'NumericalDerivativeFilter: invalid parameter window_size ({self.window_size}) changed to '
                          f'{self.poly_degree + 1}')
            self.window_size = self.poly_degree + 1

    def get_params(self):
        params_dict = {'poly_degree': self.poly_degree, 'order': self.order, 'window_size': self.window_size}
        if self.parameters_changed:
            return tuple([params_dict, self.changed_params])
        else:
            return params_dict


class CutImplementation(DataOperationImplementation):
    def __init__(self, log: Log = None, **params):
        super().__init__()
        cut_part = params.get('cut_part')

        # Define logger object
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if 0 < cut_part <= 0.9:
            self.cut_part = cut_part
            self.parameters_changed = False
        else:
            # Default parameter
            self.log.info(f"Change invalid parameter cut_part ({cut_part}) on default value (0.5)")
            self.cut_part = 0.5
            self.parameters_changed = True

    def fit(self, input_data: InputData):
        """ Class doesn't support fit operation

            :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData, is_fit_pipeline_stage: Optional[bool]) -> OutputData:
        """ Cut first cut_part from time series
            new_len = len - int(self.cut_part * (input_values.shape[0]-horizon))

            :param input_data: data with features, target and ids to process
            :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
            :return output_data: output data with cutted time series
        """
        horizon = input_data.task.task_params.forecast_length
        input_copy = copy(input_data)
        input_values = input_copy.features
        cut_len = int(self.cut_part * (input_values.shape[0] - horizon))
        output_values = input_values[cut_len::]
        if is_fit_pipeline_stage:
            input_copy.idx = np.arange(cut_len, input_values.shape[0])
        input_copy.features = output_values
        input_copy.target = output_values
        output_data = self._convert_to_output(input_copy,
                                              output_values,
                                              data_type=input_data.data_type)
        return output_data

    def get_params(self):
        params_dict = {"cut_part": self.cut_part}
        if self.parameters_changed:
            return tuple([params_dict, ['cut_part']])
        else:
            return params_dict


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


def ts_to_table(idx, time_series: np.array, window_size: int, is_lag=False):
    """ Method convert time series to lagged form.

    :param idx: the indices of the time series to convert
    :param time_series: source time series
    :param window_size: size of sliding window, which defines lag
    :param is_lag: is function used for lagged transformation.
    False needs to convert one dimensional output to lagged form.

    :return updated_idx: clipped indices of time series
    :return features_columns: lagged time series feature table
    """
    # Convert data to lagged form
    lagged_dataframe = pd.DataFrame({'t_id': time_series})
    vals = lagged_dataframe['t_id']
    for i in range(1, window_size):
        frames = [lagged_dataframe, vals.shift(i)]
        lagged_dataframe = pd.concat(frames, axis=1)

    # Remove incomplete rows
    lagged_dataframe.dropna(inplace=True)

    transformed = np.array(lagged_dataframe)

    # Generate dataset with features
    features_columns = np.fliplr(transformed)

    if is_lag:
        updated_idx = list(idx[window_size:])
        updated_idx.append(idx[-1])
        updated_idx = np.array(updated_idx)
    else:
        updated_idx = idx[:len(idx) - window_size + 1]

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


def prepare_target(all_idx, idx, features_columns: np.array, target, forecast_length: int):
    """ Method convert time series to lagged form. Transformation applied
    only for generating target table (time series considering as multi-target
    regression task).

    :param all_idx: all indices in data
    :param idx: remaining indices after lagged feature table generation
    :param features_columns: lagged feature table
    :param target: source time series
    :param forecast_length: forecast length

    :return updated_idx: clipped indices of time series
    :return updated_features: clipped lagged feature table
    :return updated_target: lagged target table
    """
    # Remove last repeated element
    idx = idx[: -1]

    # Update target (clip first "window size" values)
    row_nums = [list(all_idx).index(i) for i in idx]
    ts_target = target[row_nums]

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

        updated_idx = idx[: -forecast_length + 1]
        updated_features = features_columns[: -forecast_length]
    else:
        # Forecast horizon equals to 1
        updated_idx = idx
        updated_features = features_columns[: -1]
        updated_target = ts_target

    return updated_idx, updated_features, updated_target
