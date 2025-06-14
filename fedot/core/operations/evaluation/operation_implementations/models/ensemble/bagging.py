from typing import Optional

from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.tasks import TaskTypesEnum


def _get_prev_node(input_data: InputData):
    prev_models = input_data.supplementary_data.previous_operations
    error_message = (
        "The Bagging node requires exactly one previous model. "
        f"Got: {len(prev_models) if prev_models is not None else 'None'}"
    )

    if not prev_models:
        raise ValueError(error_message)

    if len(prev_models) > 1:
        raise ValueError(error_message)

    model = prev_models[0]
    if model is None:
        raise ValueError("The provided model is None")

    return model


class BaggingImplementation(ModelImplementation):
    """Base class for bagging operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.seed = self.params.get('seed', 42)
        self.n_estimators = self.params.get('n_estimators', 10)

        self.bagging = None
        self.classes_ = None
        self.fitted_model = None

    def _init(self, input_data: InputData):
        self.classes_ = input_data.class_labels

    def fit(self, input_data: InputData):
        self._init(input_data)

        prev_node: PipelineNode = _get_prev_node(input_data)
        est = getattr(prev_node.fitted_operation, 'model', prev_node.fitted_operation)
        model = self.bagging(
            estimator=est,
            n_estimators=self.n_estimators,
            random_state=self.seed,
        )

        self.fitted_model = model.fit(prev_node.node_data.features, prev_node.node_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        prev_node: PipelineNode = _get_prev_node(input_data)
        result = self.fitted_model.predict(prev_node.node_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=result)
        return output_data

    def predict_proba(self, input_data: InputData) -> OutputData:
        if input_data.task.task_type != TaskTypesEnum.classification:
            raise ValueError('predict_proba is only available for classification tasks')

        prev_node: PipelineNode = _get_prev_node(input_data)
        result = self.fitted_model.predict_proba(prev_node.node_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=result)
        return output_data


class FedotBaggingClassifier(BaggingImplementation):
    """Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging = BaggingClassifier


class FedotBaggingRegressor(BaggingImplementation):
    """Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging = BaggingRegressor
