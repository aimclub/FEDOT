from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import \
    FedotCatBoostClassificationImplementation, FedotCatBoostRegressionImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler


class BoostingClassificationStrategy(EvaluationStrategy):
    __operations_by_types = {
        'catboost': FedotCatBoostClassificationImplementation,
        # 'xgboost': FedotXgboostBoostClassificationImplementation,
        # 'lgbm': FedotLightGBMBoostClassificationImplementation,
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
        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['default', 'labels']:
            prediction = trained_operation.predict(predict_data)

        elif self.output_mode in ['probs', 'full_probs'] and predict_data.task:
            prediction = trained_operation.predict_proba(predict_data)

        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class BoostingRegressionStrategy(EvaluationStrategy):
    __operations_by_types = {
        'cb_regr': FedotCatBoostRegressionImplementation,
        # 'xgb_regr': FedotXgboostBoostRegressionImplementation,
        # 'lgbm_regr': FedotLightGBMBoostRegressionImplementation,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = None

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]

        else:
            raise ValueError(f'Impossible to obtain Boosting Strategy for {operation_type}')

    def fit(self, train_data: InputData):
        operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)

        return self._convert_to_output(prediction, predict_data)




