from typing import Optional

import numpy as np
import torch

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, is_multi_output_task
from fedot.core.operations.evaluation.operation_implementations.models.tabpfn import \
    FedotTabPFNClassificationImplementation, FedotTabPFNRegressionImplementation, \
    FedotAutoTabPFNClassificationImplementation, FedotAutoTabPFNRegressionImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler


class TabPFNStrategy(EvaluationStrategy):
    _operations_by_types = {
        'tabpfn': FedotTabPFNClassificationImplementation,
        'tabpfnreg': FedotTabPFNRegressionImplementation,
        'autotabpfn': FedotAutoTabPFNClassificationImplementation,
        'autotabpfnreg': FedotAutoTabPFNRegressionImplementation,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)
        self.device = params.get('device', 'auto')
        self.max_samples_cpu = params.get('max_samples_cpu', 1000)
        self.max_samples_gpu = params.get('max_samples_gpu', 5000)
        self.max_features = params.get('max_features', 500)

    def fit(self, train_data: InputData):
        check_data_size(
            data=train_data,
            device=self.device,
            max_samples_cpu=self.max_samples_cpu,
            max_samples_gpu=self.max_samples_gpu,
            max_features=self.max_features,
        )
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            raise ValueError('Time series forecasting not supported for TabPFN')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        raise NotImplementedError()


class TabPFNClassificationStrategy(TabPFNStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode == 'labels':
            output = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            n_classes = len(trained_operation.classes_)
            output = trained_operation.predict_proba(predict_data)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif (n_classes == 2 and self.output_mode != 'full_probs'
                  and len(output.predict.shape) > 1):
                output.predict = output.predict[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return output


class TabPFNRegressionStrategy(TabPFNStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        return trained_operation.predict(predict_data)


def check_data_size(
        data: InputData,
        device: str = "auto",
        max_samples_cpu: int = 1000,
        max_samples_gpu: int = 5000,
        max_features: int = 500,
) -> bool:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        max_samples = max_samples_cpu
    else:
        max_samples = max_samples_gpu

    if data.features.shape[0] > max_samples:
        raise ValueError(
            f"Input data has too many samples ({data.features.shape[0]}), "
            f"maximum is {max_samples} for device '{device}'"
        )
    if data.features.shape[1] > max_features:
        raise ValueError(
            f"Input data has too many features ({data.features.shape[1]}), "
            f"maximum is {max_features}"
        )
    return True
