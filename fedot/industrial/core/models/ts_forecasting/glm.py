from copy import copy

import numpy as np
import optuna
import statsmodels.api as sm
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from golem.core.tuning.optuna_tuner import OptunaTuner
from scipy.stats import kurtosis, skew
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian
from statsmodels.genmod.families.links import inverse_squared, log as lg
from statsmodels.genmod.generalized_linear_model import GLM

from fedot.industrial.core.repository.industrial_implementations.abstract import build_tuner


class GLMIndustrial(ModelImplementation):
    """ Generalized linear models implementation """
    family_distribution = {
        "gaussian": Gaussian(lg()),
        "gamma": Gamma(lg()),
        "inverse_gaussian": InverseGaussian(inverse_squared())
    }

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = None
        self.family_link = None
        self.auto_reg = PipelineBuilder().add_node('ar').build()
        self.ar_tuning_params = dict(
            tuner=OptunaTuner,
            metric=RegressionMetricsEnum.RMSE,
            tuning_timeout=1,
            tuning_early_stop=20,
            tuning_iterations=50)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    @property
    def family(self) -> str:
        return self.params.get('family')

    @property
    def link(self) -> str:
        return self.params.get('link')

    def _check_glm_params(self, mean_kurtosis, mean_skew):
        if np.logical_or(
            mean_kurtosis < -1,
            mean_kurtosis > 1) and np.logical_or(
                mean_skew < -0.2,
                mean_skew > 0.2):
            family = 'gamma'
        elif np.logical_or(mean_kurtosis < -2, mean_kurtosis > 2) and np.logical_or(mean_skew < -0.5, mean_skew > 0.5):
            family = "inverse_gaussian"
        else:
            family = 'gaussian'
        return family

    def fit(self, input_data):
        tuned_model = build_tuner(self,
                                  model_to_tune=self.auto_reg,
                                  tuning_params=self.ar_tuning_params,
                                  train_data=input_data)
        self.auto_reg = tuned_model
        residual = self.auto_reg.root_node.fitted_operation.autoreg.resid
        residual = np.nan_to_num(residual, nan=0, posinf=0, neginf=0)
        family = self._check_glm_params(kurtosis(residual), skew(residual))
        self.family_link = self.family_distribution[family]
        self.exog_residual = sm.add_constant(
            np.arange(0, residual.shape[0]).astype("float64")).reshape(-1, 2)
        self.model = GLM(exog=self.exog_residual, endog=residual.astype("float64").reshape(-1, 1),
                         family=self.family_link
                         ).fit(method="lbfgs")
        del self.ar_tuning_params
        return self.model

    def predict(self, input_data):
        autoreg_predict = self.auto_reg.predict(input_data)
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        input_data.idx
        if forecast_length == 1:
            predictions = self.model.predict(np.concatenate(
                [np.array([1]), input_data.idx.astype("float64")]).reshape(-1, 2))
        else:
            predictions = self.model.predict(self.exog_residual)
        predictions = predictions[-forecast_length:]
        predict = autoreg_predict.predict + predictions
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        autoreg_predict = self.auto_reg.predict(input_data)
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target
        predictions = self.model.predict(self.exog_residual)
        predictions = predictions[-forecast_length:]
        predict = autoreg_predict.predict + predictions
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
