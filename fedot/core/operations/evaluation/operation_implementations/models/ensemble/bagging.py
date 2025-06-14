import numpy as np
from typing import Optional

from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.tasks import TaskTypesEnum


def _assert_same_node_count(fitted_nodes: list, prev_nodes: list):
    if len(fitted_nodes) != len(prev_nodes):
        raise ValueError(f"Number of nodes mismatch: expected {len(prev_nodes)} nodes, got {len(fitted_nodes)}")


class BaggingImplementation(ModelImplementation):
    """Base class for bagging operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.seed = self.params.get('seed', 42)
        self.n_jobs = self.params.get('n_jobs', -1)
        self.n_estimators = self.params.get('n_estimators', 10)

        self.bagging_model = None
        self.classes_ = None
        self.fitted_models = []

    def _init(self, previous_nodes):
        self.classes_ = previous_nodes[0].node_data.class_labels

    def _fit(self, previous_nodes: list['PipelineNode']):
        for node in previous_nodes:
            est = node.fitted_operation.model if hasattr(node.fitted_operation, 'model') else node.fitted_operation
            model = self.bagging_model(
                estimator=est,
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
            model.fit(node.node_data.features, node.node_data.target)
            self.fitted_models.append(model)

    def fit(self, input_data: InputData, **kwargs):
        self._init(previous_nodes=kwargs['prev_nodes'])
        self._fit(previous_nodes=kwargs['prev_nodes'])
        return self

    def predict(self, input_data: InputData, **kwargs) -> OutputData:
        result = []
        previous_nodes = kwargs['prev_nodes']
        _assert_same_node_count(self.fitted_models, previous_nodes)

        for fitted_model, node in zip(self.fitted_models, previous_nodes):
            probs = fitted_model.predict(X=node.node_data.features)
            result.append(probs)
        result = np.hstack(result)
        output_data = self._convert_to_output(input_data=input_data, predict=result)
        return output_data

    def predict_proba(self, input_data: InputData, **kwargs) -> OutputData:
        if input_data.task.task_type != TaskTypesEnum.classification:
            raise ValueError('predict_proba is only available for classification tasks')

        result = []
        previous_nodes = kwargs['prev_nodes']
        _assert_same_node_count(self.fitted_models, previous_nodes)

        for fitted_model, node in zip(self.fitted_models, previous_nodes):
            probs = fitted_model.predict_proba(X=node.node_data.features)
            result.append(probs)
        result = np.hstack(result)
        output_data = self._convert_to_output(input_data=input_data, predict=result)
        return output_data


class FedotBaggingClassifier(BaggingImplementation):
    """Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging_model = BaggingClassifier


class FedotBaggingRegressor(BaggingImplementation):
    """Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging_model = BaggingRegressor
