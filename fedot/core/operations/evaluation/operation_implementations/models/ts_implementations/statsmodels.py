from copy import copy

import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian
from statsmodels.genmod.families.links import identity, inverse_power, inverse_squared, log as lg
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class GLMImplementation(ModelImplementation):
    """ Generalized linear models implementation """
    # some models are dropped due to instability
    family_distribution = {
        "gaussian": {'distribution': Gaussian,
                     'default_link': 'identity',
                     'available_links': {'log': lg(),
                                         'identity': identity(),
                                         'inverse_power': inverse_power()
                                         }
                     },
        "gamma": {'distribution': Gamma,
                  'default_link': 'inverse_power',
                  'available_links': {'log': lg(),
                                      'identity': identity(),
                                      'inverse_power': inverse_power()
                                      }
                  },
        "inverse_gaussian": {'distribution': InverseGaussian,
                             'default_link': 'inverse_squared',
                             'available_links': {'log': lg(),
                                                 'identity': identity(),
                                                 'inverse_squared': inverse_squared(),
                                                 'inverse_power': inverse_power()
                                                 }
                             },
        "default": Gaussian(identity())
    }

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        self.family_link = None

        self.correct_params()

    @property
    def family(self) -> str:
        return self.params.get('family')

    @property
    def link(self) -> str:
        return self.params.get('link')

    def fit(self, input_data):
        self.model = GLM(
            exog=sm.add_constant(input_data.idx.astype("float64")).reshape(-1, 2),
            endog=input_data.target.astype("float64").reshape(-1, 1),
            family=self.family_link
        ).fit(method="lbfgs")
        return self.model

    def predict(self, input_data):
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        if forecast_length == 1:
            predictions = self.model.predict(np.concatenate([np.array([1]),
                                                             input_data.idx.astype("float64")]).reshape(-1, 2))
        else:
            predictions = self.model.predict(sm.add_constant(input_data.idx.astype("float64")).reshape(-1, 2))

        start_id = old_idx[-1] - forecast_length + 1
        end_id = old_idx[-1]
        predict = predictions
        predict = np.array(predict).reshape(1, -1)
        new_idx = np.arange(start_id, end_id + 1)

        input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target
        predictions = self.model.predict(sm.add_constant(input_data.idx.astype("float64")).reshape(-1, 2))
        _, predict = ts_to_table(idx=old_idx,
                                 time_series=predictions,
                                 window_size=forecast_length)
        new_idx, target_columns = ts_to_table(idx=old_idx,
                                              time_series=target,
                                              window_size=forecast_length)

        input_data.idx = new_idx
        input_data.target = target_columns

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def set_default(self):
        """ Set default value of Family(link) """
        self.family_link = self.family_distribution['default']
        self.params.update(family='gaussian')
        self.log.info("Invalid family. Changed to default value")

    def correct_params(self):
        """ Correct params if they are not correct """
        if self.family in self.family_distribution:
            if self.link not in self.family_distribution[self.family]['available_links']:
                # get default link for distribution if current invalid
                default_link = self.family_distribution[self.family]['default_link']
                self.log.info(
                    f"Invalid link function {self.link} for {self.family}. Change to default "
                    f"link {default_link}")
                self.params.update(link=default_link)
            # if correct isn't need
            self.family_link = self.family_distribution[self.family]['distribution'](
                self.family_distribution[self.family]['available_links'][self.link]
            )
        else:
            # get default family for distribution if current invalid
            self.set_default()


class AutoRegImplementation(ModelImplementation):

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.autoreg = None
        self.actual_ts_len = None

    def fit(self, input_data):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        self.actual_ts_len = len(source_ts)

        # Correct window size parameter
        self._check_and_correct_lags(source_ts)

        lag_1 = int(self.params.get('lag_1'))
        lag_2 = int(self.params.get('lag_2'))
        self.autoreg = AutoReg(source_ts, lags=[lag_1, lag_2]).fit()
        self.actual_ts_len = input_data.idx.shape[0]

        return self.autoreg

    def predict(self, input_data):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length

        # in case in(out) sample forecasting
        self.handle_new_data(input_data)
        start_id = self.actual_ts_len
        end_id = start_id + forecast_length - 1
        predicted = self.autoreg.predict(start=start_id, end=end_id)
        predict = np.array(predicted).reshape(1, -1)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        idx = input_data.idx
        target = input_data.target
        predicted = self.autoreg.predict(start=idx[0], end=idx[-1])
        # adding nan to target as in predicted
        nan_mask = np.isnan(predicted)
        target = target.astype(float)
        target[nan_mask] = np.nan
        _, predict = ts_to_table(idx=idx,
                                 time_series=predicted,
                                 window_size=forecast_length)
        _, target_columns = ts_to_table(idx=idx,
                                              time_series=target,
                                              window_size=forecast_length)

        input_data.idx = input_data.idx[~nan_mask]
        input_data.target = target_columns
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def _check_and_correct_lags(self, time_series: np.array):
        previous_lag_1 = int(self.params.get('lag_1'))
        previous_lag_2 = int(self.params.get('lag_2'))
        max_lag = len(time_series) // 2 - 1
        new_lag_1 = self._check_and_correct_lag(max_lag, previous_lag_1)
        new_lag_2 = self._check_and_correct_lag(max_lag, previous_lag_2)
        if new_lag_1 == new_lag_2:
            new_lag_2 -= 1
        prefix = "Warning: lag of AutoRegImplementation was changed"
        if previous_lag_1 != new_lag_1:
            self.log.info(f"{prefix} from {previous_lag_1} to {new_lag_1}.")
            self.params.update(lag_1=new_lag_1)
        if previous_lag_2 != new_lag_2:
            self.log.info(f"{prefix} from {previous_lag_2} to {new_lag_2}.")
            self.params.update(lag_2=new_lag_2)

    def _check_and_correct_lag(self, max_lag: int, lag: int):
        if lag > max_lag:
            lag = max_lag
        return lag

    def handle_new_data(self, input_data: InputData):
        """
        Method to update x samples inside a model (used when we want to use old model to a new data)

        :param input_data: new input_data
        """
        if input_data.idx[0] > self.actual_ts_len:
            self.autoreg.model.endog = input_data.features[-self.actual_ts_len:]
            self.autoreg.model._setup_regressors()


class ExpSmoothingImplementation(ModelImplementation):
    """ Exponential smoothing implementation from statsmodels """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        if self.params.get("seasonal"):
            self.seasonal_periods = int(self.params.get("seasonal_periods"))
        else:
            self.seasonal_periods = None

    def fit(self, input_data):
        self.model = ETSModel(
            input_data.features.astype("float64"),
            error=self.params.get("error"),
            trend=self.params.get("trend"),
            seasonal=self.params.get("seasonal"),
            damped_trend=self.params.get("damped_trend"),
            seasonal_periods=self.seasonal_periods
        )
        self.model = self.model.fit(disp=False)
        return self.model

    def predict(self, input_data):
        input_data = copy(input_data)
        idx = input_data.idx

        start_id = idx[0]
        end_id = idx[-1]
        predictions = self.model.predict(start=start_id,
                                         end=end_id)
        predict = predictions
        predict = np.array(predict).reshape(1, -1)
        new_idx = np.arange(start_id, end_id + 1)

        input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        idx = input_data.idx
        target = input_data.target

        # Indexing for statsmodels is different
        start_id = idx[0]
        end_id = idx[-1]
        predictions = self.model.predict(start=start_id,
                                         end=end_id)
        _, predict = ts_to_table(idx=idx,
                                 time_series=predictions,
                                 window_size=forecast_length)
        new_idx, target_columns = ts_to_table(idx=idx,
                                              time_series=target,
                                              window_size=forecast_length)

        input_data.idx = new_idx
        input_data.target = target_columns

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data
