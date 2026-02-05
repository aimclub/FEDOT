import json
from copy import deepcopy, copy
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.tuning.search_space import get_industrial_search_space
from fedot.industrial.tools.serialisation.path_lib import PATH_TO_DEFAULT_PARAMS


class LaggedAR(ModelImplementation):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.channel_model = self.params.get("channel_model", "ridge")
        self.window_size = self.params.get("window_size", 10)
        self.tuning_params = dict(
            tuner=SimultaneousTuner,
            metric=self.params.get("metric", RegressionMetricsEnum.RMSE),
            tuning_timeout=1,
            tuning_early_stop=10,
            tuning_iterations=20,
        )

    def _define_model(self):
        with open(PATH_TO_DEFAULT_PARAMS) as json_data:
            self.default_operation_params = json.load(json_data)
        self.default_channel_model_params = self.default_operation_params[self.channel_model]
        self.model = PipelineBuilder().add_node(self.channel_model,
                                                params=self.default_channel_model_params).build()
        return self.model

    def _define_search_space(self):
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(
            custom_search_space=custom_search_space, replace_default_search_space=True
        )

        return search_space

    def _define_tuning_data(self, train_data):
        tuning_data = deepcopy(train_data)
        tuning_data.data_type = DataTypesEnum.table
        tuning_data.task.task_type = TaskTypesEnum.ts_forecasting
        return tuning_data

    def _create_tuner(self, search_space, tuning_params, tuning_data):
        return (
            TunerBuilder(tuning_data.task)
            .with_search_space(search_space)
            .with_tuner(tuning_params["tuner"])
            .with_n_jobs(1)
            .with_metric(tuning_params["metric"])
            .with_iterations(tuning_params.get("tuning_iterations", 20))
            .build(tuning_data)
        )

    def build_tuner(self, model_to_tune, tuning_params, train_data):
        tuning_data = self._define_tuning_data(train_data)
        search_space = self._define_search_space()
        pipeline_tuner = self._create_tuner(search_space, tuning_params, tuning_data)
        model_to_tune = pipeline_tuner.tune(model_to_tune)
        model_to_tune.fit(train_data)
        del pipeline_tuner
        return model_to_tune

    def _create_pcd(self, input_data, is_fit_stage: bool = True):
        train_data = copy(input_data)

        input_data.features = HankelMatrix(
            time_series=train_data.features,
            window_size=self.ts_patch_len).trajectory_matrix.T
        if is_fit_stage:
            input_data.target = HankelMatrix(
                time_series=train_data.features[self.ts_patch_len:],
                window_size=train_data.task.task_params.forecast_length).trajectory_matrix.T
            input_data.features = input_data.features[:input_data.target.shape[0], :]
        else:
            if input_data.target is not None:
                if len(input_data.target.shape) < 2:
                    lagged_target = HankelMatrix(
                        time_series=train_data.features[self.ts_patch_len:],
                        window_size=train_data.task.task_params.forecast_length).trajectory_matrix.T
                    input_data.features = input_data.features[:lagged_target.shape[0], :]
                    input_data.target = lagged_target
        return input_data

    def fit(self, input_data):
        self.ts_patch_len = round(input_data.features.shape[0] * 0.01 * self.window_size)
        input_data = self._create_pcd(input_data, True)
        model_to_tune = self._define_model()
        self.tuned_model = self.build_tuner(
            model_to_tune=model_to_tune,
            tuning_params=self.tuning_params,
            train_data=input_data)

        del self.tuning_params
        return self

    def _predict(self, input_data: InputData) -> OutputData:
        input_data = self._create_pcd(input_data, False)
        prediction = self.tuned_model.predict(input_data)

        forecast_length = input_data.task.task_params.forecast_length

        # Calculate n_rows based on actual array size to ensure exact division
        if prediction.predict.shape[0] % forecast_length != 0:
            # Trim the prediction array to make it divisible by forecast_length
            trimmed_size = (prediction.predict.shape[0] // forecast_length) * forecast_length
            if trimmed_size == 0:
                raise ValueError(
                    "Forecast length is greater than the number of predictions, resulting in an invalid trimmed size.")
            prediction.predict = prediction.predict[:trimmed_size, ...]

        n_rows = prediction.predict.size // forecast_length
        n_cols = forecast_length

        prediction.predict = prediction.predict.reshape(n_rows, n_cols)
        return prediction

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)
