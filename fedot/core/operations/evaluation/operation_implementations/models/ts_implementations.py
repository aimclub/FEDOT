from typing import Optional

import numpy as np
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA

from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import _ts_to_table
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class ARIMAImplementation(ModelImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.arima = None
        self.lmbda = None
        self.scope = None
        self.actual_ts_len = None
        self.sts = None
        # TODO for some configuration of p,d,q got ValueError

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

        transformed_ts, self.lmbda = stats.boxcox(source_ts)

        if not self.params:
            # Default data
            self.params = {'p': 2, 'd': 0, 'q': 2}

        p = int(self.params.get('p'))
        d = int(self.params.get('d'))
        q = int(self.params.get('q'))
        params = {'order': (p, d, q)}
        self.arima = ARIMA(transformed_ts, **params).fit()

        return self.arima

    def predict(self, input_data, is_fit_chain_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with smoothed time series
        """
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        # For training chain get fitted data
        if is_fit_chain_stage:
            fitted_values = self.arima.fittedvalues

            fitted_values = self._inverse_boxcox(predicted=fitted_values,
                                                 lmbda=self.lmbda)
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
                                             lmbda=self.lmbda)

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

    @staticmethod
    def _inverse_boxcox(predicted, lmbda):
        """ Method apply inverse Box-Cox transformation """
        if lmbda == 0:
            return np.exp(predicted)
        else:
            return np.exp(np.log(lmbda * predicted + 1) / lmbda)

    def _inverse_shift(self, values):
        """ Method apply inverse shift operation """
        if self.scope is None:
            pass
        else:
            values = values - self.scope

        return values


class AutoRegImplementation(ModelImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.actual_ts_len = None
        self.autoreg = None

    def fit(self, input_data):
        """ Class fit arima model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        self.actual_ts_len = len(source_ts)

        if not self.params:
            # Default data
            self.params = {'lag_1': 12, 'lag_2': 60}

        lag_1 = int(self.params.get('lag_1'))
        lag_2 = int(self.params.get('lag_2'))
        params = {'lags': [lag_1, lag_2]}
        self.autoreg = AutoReg(source_ts, **params).fit()

        return self.autoreg

    def predict(self, input_data, is_fit_chain_stage: bool):
        """ Method for smoothing time series

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with smoothed time series
        """
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        if is_fit_chain_stage:
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
