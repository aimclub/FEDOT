from typing import Optional

import numpy as np

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.tensor_tabular import (
    FedotFTTransformerClassificationImplementation,
    FedotFTTransformerRegressionImplementation,
    FedotRealMLPClassificationImplementation,
    FedotRealMLPRegressionImplementation,
    FedotResNetClassificationImplementation,
    FedotResNetRegressionImplementation,
    FedotTabMClassificationImplementation,
    FedotTabMRegressionImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler


class TensorTabularStrategy(EvaluationStrategy):
    _operations_by_types = {}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation


class TensorTabularClassificationStrategy(TensorTabularStrategy):
    _operations_by_types = {
        'tabm': FedotTabMClassificationImplementation,
        'ft_transformer': FedotFTTransformerClassificationImplementation,
        'tab_resnet': FedotResNetClassificationImplementation,
        'realmlp': FedotRealMLPClassificationImplementation,
    }

    def predict(self, trained_operation, predict_data: TensorData):
        if self.output_mode == 'labels':
            return trained_operation.predict(predict_data)
        if self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
            proba = prediction.predict
            n_classes = len(trained_operation.classes_)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            if n_classes == 2:
                if proba.ndim == 1 and self.output_mode == 'full_probs':
                    proba = np.vstack((1 - proba, proba)).T
                elif proba.ndim > 1 and self.output_mode != 'full_probs':
                    proba = proba[:, 1]
            prediction.predict = proba
            return prediction
        raise ValueError(f'Output mode {self.output_mode} is not supported')


class TensorTabularRegressionStrategy(TensorTabularStrategy):
    _operations_by_types = {
        'tabmreg': FedotTabMRegressionImplementation,
        'ft_transformerreg': FedotFTTransformerRegressionImplementation,
        'tab_resnetreg': FedotResNetRegressionImplementation,
        'realmlpreg': FedotRealMLPRegressionImplementation,
    }

    def predict(self, trained_operation, predict_data: TensorData):
        if predict_data.task.task_type is not TaskTypesEnum.regression:
            raise ValueError('Tensor tabular regression model supports only regression task')
        return trained_operation.predict(predict_data)
