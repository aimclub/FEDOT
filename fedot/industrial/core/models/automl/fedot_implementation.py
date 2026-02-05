from typing import Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters

from fedot.industrial.core.architecture.abstraction.client import use_default_fedot_client
from fedot.industrial.core.repository.model_repository import default_industrial_availiable_operation


class FedotAutomlImplementation(ModelImplementation):
    """Implementation of Fedot as classification pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'classification')

    def __init__(self, params: Optional[OperationParameters] = None):
        if not params:
            params = OperationParameters()
        else:
            params = params.to_dict()
        if 'available_operations' not in params.keys():
            params.update({'available_operations': self.AVAILABLE_OPERATIONS})
        self.model = Fedot(**params)
        super(FedotAutomlImplementation, self).__init__()

    def fit(self, input_data: InputData):
        self.model.fit(input_data)
        return self

    def predict(
            self,
            input_data: InputData,
            output_mode='default') -> OutputData:
        return self.model.current_pipeline.predict(
            input_data, output_mode=output_mode)


class FedotClassificationImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as classification pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'classification')


class FedotRegressionImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as regression pipeline node for AutoML.

    """
    AVAILABLE_OPERATIONS = default_industrial_availiable_operation(
        'regression')


class FedotForecastingImplementation(FedotAutomlImplementation):
    """Implementation of Fedot as forecasting pipeline node for AutoML.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.model = Fedot
        self.metric = params.get('metric', 'mape')
        self.timeout = params.get('timeout', 5)
        self.finetune = params.get('with_tuning', True)
        self.available_operations = ['ar',
                                     'gaussian_filter',
                                     'lagged',
                                     'lasso',
                                     'rfr',
                                     'ridge',
                                     'sgdr',
                                     'smoothing',
                                     'sparse_lagged',
                                     'svr'
                                     ]

    @use_default_fedot_client
    def fit(self, input_data: InputData):
        self.model = self.model(task_params=input_data.task.task_params,
                                problem='ts_forecasting',
                                available_operations=self.available_operations,
                                metric=self.metric,
                                with_tuning=self.finetune,
                                logging_level=30,
                                timeout=self.timeout)
        self.model.fit(input_data)
        self.model = self.model.current_pipeline
        return self

    def predict(
            self,
            input_data: InputData,
            output_mode='default') -> OutputData:
        return self.model.predict(input_data)
