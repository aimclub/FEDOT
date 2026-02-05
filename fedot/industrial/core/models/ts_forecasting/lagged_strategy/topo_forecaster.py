from copy import deepcopy
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    LaggedTransformationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot.industrial.core.models.ts_forecasting.lagged_strategy.lagged_forecaster import LaggedAR
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix


class TopologicalAR(LaggedAR):
    """Implementation of a composite of a topological extractor and autoreg atomized models"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.ts_patch_len = self.params.get("patch_len", 10)
        self.lagged_node = LaggedTransformationImplementation(OperationParameters(window_size=10))
        self.topo_ts = PipelineBuilder().add_node('topological_extractor', params={
            'window_size': self.window_size})

    def _create_pcd(self, input_data):
        new_input_data = self._convert_input_data(deepcopy(input_data))
        ts_patch_len = round(input_data.features.shape[0] * 0.01 * self.ts_patch_len)
        input_data.features = HankelMatrix(
            time_series=new_input_data.features,
            window_size=ts_patch_len).trajectory_matrix.T
        input_data.target = HankelMatrix(
            time_series=new_input_data.features[ts_patch_len:],
            window_size=new_input_data.task.task_params.forecast_length).trajectory_matrix.T
        input_data.features = input_data.features[:input_data.target.shape[0], :]
        return input_data

    def fit(self, input_data):
        input_data = self._create_pcd(input_data)
        self.topo_ts = self.topo_ts.build()
        input_data.features = self.topo_ts.fit(input_data).predict.squeeze()
        self.num_features = input_data.features.shape[1]
        self.tuned_model = self.build_tuner(
            model_to_tune=PipelineBuilder().add_node(self.channel_model).build(),
            tuning_params=self.tuning_params,
            train_data=input_data)

        del self.tuning_params
        return self

    def _predict(self, input_data: InputData) -> OutputData:
        input_data = self._create_pcd(input_data)
        flatten_prediction = self.topo_ts.predict(input_data).predict
        table_prediction = flatten_prediction.reshape(int(flatten_prediction.shape[0] / self.num_features),
                                                      self.num_features)
        input_data.features = table_prediction
        prediction = self.tuned_model.predict(input_data)
        prediction.predict = prediction.predict[-input_data.task.task_params.forecast_length:]
        return prediction

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def predict(self, input_data: InputData) -> OutputData:
        return self._predict(input_data)

    def _convert_input_data(self, input_data: InputData) -> InputData:
        if len(input_data.target.shape) < 2:
            return self.lagged_node.transform_for_fit(input_data)
        return input_data
