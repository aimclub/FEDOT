from copy import copy
from typing import Optional

import numpy as np
from statsmodels.genmod.families import Gaussian, Gamma, InverseGaussian, Poisson, Tweedie
from statsmodels.genmod.families.links import log as lg, identity, sqrt, inverse_power, inverse_squared, Power
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ts_to_table
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum
import statsmodels.api as sm


class GLMImplementation(ModelImplementation):
    """ Generalized linear models implementation """
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
        "poisson": {'distribution': Poisson,
                    'default_link': 'log',
                    'available_links': {'log': lg(),
                                        'identity': identity(),
                                        'sqrt': sqrt()
                                        }
                    },
        "tweedie": {'distribution': Tweedie,
                    'default_link': 'log',
                    'available_links': {'log': lg(),
                                        'power': Power(),
                                        }
                    },
        "default": Gaussian(identity())
    }

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.model = None
        self.params = params

        self.family_link = None

        self.params_changed = False

        self.family = self.params.get('family')
        self.link = self.params.get('link')

        self.correct_params()

    def fit(self, input_data):
        self.model = GLM(
            exog=sm.add_constant(input_data.idx.astype("float64")).reshape(-1, 2),
            endog=input_data.target.astype("float64").reshape(-1, 1),
            family=self.family_link
        ).fit()
        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target
        if forecast_length == 1 and not is_fit_pipeline_stage:
            predictions = self.model.predict(np.concatenate([np.array([1]),
                                                             input_data.idx.astype("float64")]).reshape(-1, 2))
        else:
            predictions = self.model.predict(sm.add_constant(input_data.idx.astype("float64")).reshape(-1, 2))

        if is_fit_pipeline_stage:
            _, predict = ts_to_table(idx=old_idx,
                                     time_series=predictions,
                                     window_size=forecast_length)
            new_idx, target_columns = ts_to_table(idx=old_idx,
                                                  time_series=target,
                                                  window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predict = predictions
            predict = np.array(predict).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        params_dict = {'family': self.family, 'link': self.link}
        changed_params = ['family', 'link']
        if changed_params:
            return tuple([params_dict, changed_params])
        else:
            return params_dict

    def set_default(self):
        """ Set default value of Family(link) """
        self.family_link = self.family_distribution['default']
        self.params_changed = True
        self.family = 'gaussian'
        self.log.info(
            f"Invalid family. Changed to default value")

    def correct_params(self):
        """ Correct params if they are not correct """
        if self.family in self.family_distribution:
            self.family = self.family
            if self.link not in self.family_distribution[self.family]['available_links']:
                # get default link for distribution if current invalid
                self.log.info(
                    f"Invalid link function {self.link} for {self.family}. Change to default "
                    f"link {self.family_distribution[self.family]['default_link']}")
                self.link = self.family_distribution[self.family]['default_link']
                self.params_changed = True
            # if correct isn't need
            self.family_link = self.family_distribution[self.family]['distribution'](
                self.family_distribution[self.family]['available_links'][self.link]
            )
        else:
            # get default family for distribution if current invalid
            self.set_default()


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
            predicted = self.autoreg.predict(start=old_idx[0], end=old_idx[-1])
            _, predict = ts_to_table(idx=old_idx,
                                     time_series=predicted,
                                     window_size=forecast_length)
            new_idx, target_columns = ts_to_table(idx=old_idx,
                                                  time_series=target,
                                                  window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.autoreg.predict(start=start_id,
                                             end=end_id)
            predict = np.array(predicted).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params


class ExpSmoothingImplementation(ModelImplementation):
    """ Exponential smoothing implementation from statsmodels """

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.model = None
        self.params = params
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

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        if is_fit_pipeline_stage:
            # Indexing for statsmodels is different
            predictions = self.model.predict(start=old_idx[0] + 1,
                                             end=old_idx[-1] + 1)
            _, predict = ts_to_table(idx=old_idx,
                                     time_series=predictions,
                                     window_size=forecast_length)
            new_idx, target_columns = ts_to_table(idx=old_idx,
                                                  time_series=target,
                                                  window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        else:
            start_id = old_idx[-1] - forecast_length + 2
            end_id = old_idx[-1] + 1
            predictions = self.model.predict(start=start_id,
                                             end=end_id)
            predict = predictions
            predict = np.array(predict).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params
