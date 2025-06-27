from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import \
    FedotCatBoostClassificationImplementation, FedotCatBoostRegressionImplementation, \
    FedotXGBoostClassificationImplementation, FedotXGBoostRegressionImplementation, \
    FedotLightGBMClassificationImplementation, FedotLightGBMRegressionImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.evaluation_interfaces import is_multi_output_task
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler


class BoostingStrategy(EvaluationStrategy):
    __operations_by_types = {
        'catboost': FedotCatBoostClassificationImplementation,
        'catboostreg': FedotCatBoostRegressionImplementation,
        'xgboost': FedotXGBoostClassificationImplementation,
        'xgboostreg': FedotXGBoostRegressionImplementation,
        'lgbm': FedotLightGBMClassificationImplementation,
        'lgbmreg': FedotLightGBMRegressionImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]

        else:
            raise ValueError(f'Impossible to obtain Boosting Strategy for {operation_type}')

    def fit(self, train_data: InputData):
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            raise ValueError('Time series forecasting not supported for boosting models')

        if is_multi_output_task(train_data):
            if self.operation_type == 'catboost':
                self.params_for_fit.update(loss_function='MultiLogloss')
            elif self.operation_type == 'catboostreg':
                self.params_for_fit.update(loss_function='MultiRMSE')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        raise NotImplementedError()


class BoostingClassificationStrategy(BoostingStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['labels']:
            prediction = trained_operation.predict(predict_data)
        elif (self.output_mode in ['probs', 'full_probs', 'default'] and
              predict_data.task.task_type is TaskTypesEnum.classification):
            n_classes = len(trained_operation.classes_)
            is_multi_output_target = is_multi_output_task(predict_data)

            prediction = trained_operation.predict_proba(predict_data)
            if isinstance(prediction, OutputData):
                prediction = prediction.predict
            is_prediction_correct = self._check_prediction_correctness(prediction)

            if n_classes < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs' and is_prediction_correct:
                if is_multi_output_target and isinstance(prediction, list):
                    prediction = np.stack([pred[:, 1] for pred in prediction]).T
                else:
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output mode {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)

    @staticmethod
    def _check_prediction_correctness(prediction) -> bool:
        if isinstance(prediction, list):
            return len(prediction[0].shape) > 1
        else:
            return len(prediction.shape) > 1


class BoostingRegressionStrategy(BoostingStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)

        return self._convert_to_output(prediction, predict_data)
