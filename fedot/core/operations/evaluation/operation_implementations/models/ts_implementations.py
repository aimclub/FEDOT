from typing import Optional
from copy import copy

import numpy as np
from scipy import stats
from scipy.special import inv_boxcox, boxcox
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import _ts_to_table
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.utilities.ts_gapfilling import SimpleGapFiller


class ARIMAImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.params = params
        self.arima = None
        self.lambda_value = None
        self.scope = None
        self.actual_ts_len = None
        self.sts = None

    def fit(self, input_data):
        """ Class fit arima model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        # Save actual time series length
        self.actual_ts_len = len(source_ts)
        self.sts = source_ts

        # Apply box-cox transformation for positive values
        min_value = np.min(source_ts)
        if min_value > 0:
            pass
        else:
            # Making a shift to positive values
            self.scope = abs(min_value) + 1
            source_ts = source_ts + self.scope

        _, self.lambda_value = stats.boxcox(source_ts)
        transformed_ts = boxcox(source_ts, self.lambda_value)

        # Set parameters
        p = int(self.params.get('p'))
        d = int(self.params.get('d'))
        q = int(self.params.get('q'))
        params = {'order': (p, d, q)}
        self.arima = ARIMA(transformed_ts, **params).fit()

        return self.arima

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        # For training pipeline get fitted data
        if is_fit_pipeline_stage:
            fitted_values = self.arima.fittedvalues

            fitted_values = self._inverse_boxcox(predicted=fitted_values,
                                                 lambda_param=self.lambda_value)
            # Undo shift operation
            fitted_values = self._inverse_shift(fitted_values)

            diff = int(self.actual_ts_len - len(fitted_values))
            # If first elements skipped
            if diff != 0:
                # Fill nans with first values
                first_element = fitted_values[0]
                first_elements = [first_element] * diff
                first_elements.extend(list(fitted_values))

                fitted_values = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted_values,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.arima.predict(start=start_id,
                                           end=end_id)

            predicted = self._inverse_boxcox(predicted=predicted,
                                             lambda_param=self.lambda_value)

            # Undo shift operation
            predict = self._inverse_shift(predicted)
            # Convert one-dim array as column
            predict = np.array(predict).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params

    def _inverse_boxcox(self, predicted, lambda_param):
        """ Method apply inverse Box-Cox transformation """
        if lambda_param == 0:
            return np.exp(predicted)
        else:
            res = inv_boxcox(predicted, lambda_param)
            res = self._filling_gaps(res)
            return res

    def _inverse_shift(self, values):
        """ Method apply inverse shift operation """
        if self.scope is None:
            pass
        else:
            values = values - self.scope

        return values

    @staticmethod
    def _filling_gaps(res):
        nan_ind = np.argwhere(np.isnan(res))
        res[nan_ind] = -100.0

        # Gaps in first and last elements fills with mean value
        if 0 in nan_ind:
            res[0] = np.mean(res)
        if int(len(res) - 1) in nan_ind:
            res[int(len(res) - 1)] = np.mean(res)

        # Gaps in center of timeseries fills with linear interpolation
        if len(np.ravel(np.argwhere(np.isnan(res)))) != 0:
            gf = SimpleGapFiller()
            res = gf.linear_interpolation(res)

        return res


class AutoRegImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.params = params
        self.actual_ts_len = None
        self.autoreg = None

    def fit(self, input_data):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        self.actual_ts_len = len(source_ts)
        lag_1 = int(self.params.get('lag_1'))
        lag_2 = int(self.params.get('lag_2'))
        params = {'lags': [lag_1, lag_2]}
        self.autoreg = AutoReg(source_ts, **params).fit()

        return self.autoreg

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        if is_fit_pipeline_stage:
            fitted = self.autoreg.predict(start=old_idx[0], end=old_idx[-1])
            # First n elements in time series are skipped
            diff = self.actual_ts_len - len(fitted)

            # Fill nans with first values
            first_element = fitted[0]
            first_elements = [first_element] * diff
            first_elements.extend(list(fitted))

            fitted = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.autoreg.predict(start=start_id,
                                             end=end_id)

            # Convert one-dim array as column
            predict = np.array(predicted).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

            # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params


class STLForecastARIMAImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = params
        self.model = None
        self.lambda_param = None
        self.scope = None
        self.actual_ts_len = None
        self.sts = None

    def fit(self, input_data):
        """ Class fit STLForecast arima model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        # Save actual time series length
        self.actual_ts_len = len(source_ts)
        self.sts = source_ts

        if not self.params:
            # Default data
            self.params = {'p': 2, 'd': 0, 'q': 2, 'period': 365}

        p = int(self.params.get('p'))
        d = int(self.params.get('d'))
        q = int(self.params.get('q'))
        period = int(self.params.get('period'))
        params = {'period': period, 'model_kwargs': {'order': (p, d, q)}}
        self.model = STLForecast(source_ts, ARIMA, **params).fit()

        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        # For training pipeline get fitted data
        if is_fit_pipeline_stage:
            fitted_values = self.model.get_prediction(start=old_idx[0], end=old_idx[-1]).predicted_mean
            diff = int(self.actual_ts_len) - len(fitted_values)
            # If first elements skipped
            if diff != 0:
                # Fill nans with first values
                first_element = fitted_values[0]
                first_elements = [first_element] * diff
                first_elements.extend(list(fitted_values))

                fitted_values = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted_values,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.model.get_prediction(start=start_id, end=end_id).predicted_mean

            # Convert one-dim array as column
            predict = np.array(predicted).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params
