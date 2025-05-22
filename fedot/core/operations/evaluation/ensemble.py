from typing import Optional
from abc import abstractmethod

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.bagging import \
    CatBoostBaggingClassification, CatBoostBaggingRegression, XGBoostBaggingClassification, XGBoostBaggingRegression, \
    LGBMBaggingClassification, LGBMBaggingRegression, RFBaggingClassification, RFBaggingRegression
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.blending import (
    BlendingClassifier, BlendingRegressor)
from fedot.core.operations.evaluation.operation_implementations.models.ensemble.stacking import (
    StackingClassifier, StackingRegressor)
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.evaluation_interfaces import is_multi_output_task
from fedot.utilities.random import ImplementationRandomStateHandler
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class EnsembleStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the ensemble methods

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``blending`` -> BlendingClassifier
                - ``blendreg`` -> BlendingRegressor
                - ``stacking`` -> StackingClassifier
                - ``stackreg`` -> StackingRegressor

        params: hyperparameters to fit the operation with
    """
    _operations_by_types = {
        'blending': BlendingClassifier,
        'blendreg': BlendingRegressor,
        'stacking': StackingClassifier,
        'stackreg': StackingRegressor,
        'cb_bag': CatBoostBaggingClassification,
        'cbreg_bag': CatBoostBaggingRegression,
        'xgb_bag': XGBoostBaggingClassification,
        'xgbreg_bag': XGBoostBaggingRegression,
        'lgbm_bag': LGBMBaggingClassification,
        'lgbmreg_bag': LGBMBaggingRegression,
        'rf_bag': RFBaggingClassification,
        'rfr_bag': RFBaggingRegression,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        if is_multi_output_task(train_data):
            raise ValueError(f'Ensemble methods temporary do not support multi-output tasks.')

        operation_implementation = self.operation_impl(self.params_for_fit)

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)

        return operation_implementation

    @abstractmethod
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """This method used for prediction of the target data

        Args:
            trained_operation: operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        raise AbstractMethodNotImplementError


class EnsembleClassificationStrategy(EnsembleStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
        else:
            raise ValueError(f'Output mode {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class EnsembleRegressionStrategy(EnsembleStrategy):
    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)
        return self._convert_to_output(prediction, predict_data)
