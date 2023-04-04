from copy import copy

import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian
from statsmodels.genmod.families.links import identity, inverse_power, inverse_squared, log as lg
from statsmodels.genmod.generalized_linear_model import GLM

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
