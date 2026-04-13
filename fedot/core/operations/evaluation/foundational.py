import importlib.util
import logging
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot.core.operations.evaluation.operation_implementations.models.foundational import \
    FedotTabICLClassificationImplementation, FedotTabICLRegressionImplementation, \
    FedotTabPFNClassificationImplementation, FedotTabPFNRegressionImplementation


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def check_data_size(
        data: InputData,
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

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)
        self.device = params.get('device', 'auto') if params else 'auto'
        self.max_samples = params.get('max_samples', 1000) if params else 1000
        self.max_features = params.get('max_features', 500) if params else 500

    @property
    def _is_available(self) -> bool:
        return is_package_available(self._package_name)

    def _raise_if_not_available(self):
        if not self._is_available:
            logging.log(100, self._error_message)
            raise ModuleNotFoundError(self._error_message)

    def fit(self, train_data: InputData):
        self._raise_if_not_available()

        check_data_size(
            data=train_data,
            device=self.device,
            max_samples=self.max_samples,
            max_features=self.max_features,
        )
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            raise ValueError(f'Time series forecasting not supported for {self._model_name}')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        raise NotImplementedError()


class BaseFoundationalClassificationStrategy(BaseFoundationalStrategy):
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        self._raise_if_not_available()

        if self.output_mode == 'labels':
            output = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            n_classes = len(trained_operation.classes_)
            output = trained_operation.predict_proba(predict_data)
            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            if n_classes == 2 and self.output_mode != 'full_probs' and len(output.predict.shape) > 1:
                output.predict = output.predict[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return output


class BaseFoundationalRegressionStrategy(BaseFoundationalStrategy):
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        self._raise_if_not_available()
        return trained_operation.predict(predict_data)


class TabPFNStrategy(BaseFoundationalStrategy):
    _operations_by_types = {
        'tabpfn': FedotTabPFNClassificationImplementation,
        'tabpfnreg': FedotTabPFNRegressionImplementation,
        'tabpfn_gpu': FedotTabPFNClassificationImplementation,
        'tabpfnreg_gpu': FedotTabPFNRegressionImplementation,
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
    _operations_by_types = {
        'tabicl': FedotTabICLClassificationImplementation,
        'tabiclreg': FedotTabICLRegressionImplementation,
        'tabicl_gpu': FedotTabICLClassificationImplementation,
        'tabiclreg_gpu': FedotTabICLRegressionImplementation,
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
