import importlib.util
import logging
from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.operation_execution import prepare_device_params
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot.core.operations.implementations.models.foundation import (
    FedotTabICLClassificationImplementation,
    FedotTabICLRegressionImplementation,
    FedotTabPFNClassificationImplementation,
    FedotTabPFNRegressionImplementation,
)


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def check_data_size(
        data: TensorData,
        device: str = "auto",
        max_samples: int = 1000,
        max_features: int = 500,
) -> bool:
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


class BaseFoundationalStrategy(EvaluationStrategy):
    _operations_by_types = {}
    _package_name = ''
    _model_name = ''
    _error_message = ''
    _default_max_samples = {}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def prepare_execution_params(self, params, device):
        params = prepare_device_params(params, device)
        if params.get('max_samples') is None:
            params.update(max_samples=self._default_max_samples[device.type])
        return params

    @property
    def _is_available(self) -> bool:
        return is_package_available(self._package_name)

    def _raise_if_not_available(self):
        if not self._is_available:
            logging.log(100, self._error_message)
            raise ModuleNotFoundError(self._error_message)

    def fit(self, train_data: TensorData):
        self._raise_if_not_available()
        max_samples = self.params_for_fit.get('max_samples')
        if max_samples is None:
            max_samples = self._default_max_samples['cuda' if train_data.device.type == 'cuda' else 'cpu']
        check_data_size(
            data=train_data,
            device=self.params_for_fit.get('device', 'auto'),
            max_samples=max_samples,
            max_features=self.params_for_fit.get('max_features', 500),
        )
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            raise ValueError(f'Time series forecasting not supported for {self._model_name}')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        raise NotImplementedError()


class BaseFoundationalClassificationStrategy(BaseFoundationalStrategy):
    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        self._raise_if_not_available()

        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
            if len(trained_operation.classes_) < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            if prediction.shape[-1] == 2 and self.output_mode != 'full_probs':
                prediction = prediction[..., 1]
        else:
            raise ValueError(f'Output mode {self.output_mode!r} is not supported')
        return self._replace_predict_in_tensor_data(prediction, predict_data)


class BaseFoundationalRegressionStrategy(BaseFoundationalStrategy):
    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        self._raise_if_not_available()
        prediction = trained_operation.predict(predict_data)
        return self._replace_predict_in_tensor_data(prediction, predict_data)


class TabPFNStrategy(BaseFoundationalStrategy):
    _default_max_samples = {'cpu': 1000, 'cuda': 10000}
    _operations_by_types = {
        'tabpfn': FedotTabPFNClassificationImplementation,
        'tabpfnreg': FedotTabPFNRegressionImplementation,
    }
    _package_name = 'tabpfn'
    _model_name = 'TabPFN'
    _error_message = (
        "TabPFN is required but not installed. "
        "Install with `pip install fedot[extra]` or `pip install tabpfn`."
    )


class TabPFNClassificationStrategy(TabPFNStrategy, BaseFoundationalClassificationStrategy):
    pass


class TabPFNRegressionStrategy(TabPFNStrategy, BaseFoundationalRegressionStrategy):
    pass


class TabICLStrategy(BaseFoundationalStrategy):
    _default_max_samples = {'cpu': 5000, 'cuda': 30000}
    _operations_by_types = {
        'tabicl': FedotTabICLClassificationImplementation,
        'tabiclreg': FedotTabICLRegressionImplementation,
    }
    _package_name = 'tabicl'
    _model_name = 'TabICL'
    _error_message = (
        "TabICL is required but not installed. "
        "Install with `pip install fedot[extra]` or `pip install tabicl`."
    )


class TabICLClassificationStrategy(TabICLStrategy, BaseFoundationalClassificationStrategy):
    pass


class TabICLRegressionStrategy(TabICLStrategy, BaseFoundationalRegressionStrategy):
    pass
