from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression as linreg

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


BAGGING_METHOD = {
    'mean': np.mean,
    'median': np.median,
    'max': np.max,
    'min': np.min,
    'weighted': linreg
}


class BaggingEnsemble(DataOperationImplementation):
    """Class ensemble predictions.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.bagging_method = params.get('method', 'weighted')
        self.stride = params.get('stride', 1)
        self.add_global_features = params.get('add_global_features', True)
        self.is_linreg_ensemble = self.bagging_method.__contains__('weighted')

    def _preprocess_predict_for_task(self, input_data):
        self.is_clf_task = input_data.task.task_type == TaskTypesEnum.classification
        is_forecasting_task = input_data.task.task_type == TaskTypesEnum.ts_forecasting
        is_target_table = len(input_data.target.shape) > 1
        self.is_clf_task_with_regression_ensemble = all([self.is_linreg_ensemble,
                                                         self.is_clf_task])
        is_forecasting_task_with_lagged_target = all([is_target_table,
                                                      is_forecasting_task])
        if is_forecasting_task_with_lagged_target:
            # take last column from target (horizon)
            input_data.target = input_data.target[-1]
        if self.is_clf_task_with_regression_ensemble:
            # concatenate probs output from different models
            input_data.features = input_data.features.reshape(input_data.features.shape[0], -1)

        return input_data

    def fit(self, input_data: InputData):
        """ Method fit model on a dataset

        :param input_data: data with features, target and ids to process
        """
        input_data = self._preprocess_predict_for_task(input_data)
        if self.is_linreg_ensemble:
            self.method_impl = BAGGING_METHOD[self.bagging_method]()
            self.method_impl.fit(input_data.features, input_data.target)
        else:
            self.method_impl = BAGGING_METHOD[self.bagging_method]
        return self

    def transform(self, input_data: InputData) -> OutputData:
        """ Method make prediction

        :param input_data: data with features, target and ids to process
        """
        input_data = self._preprocess_predict_for_task(input_data)
        if not self.is_linreg_ensemble:
            ensembled_predict = self.method_impl(input_data.features, axis=1)
        else:
            ensembled_predict = self.method_impl.predict(input_data.features)
        ensembled_predict = np.round(abs(ensembled_predict)) \
            if self.is_clf_task_with_regression_ensemble else ensembled_predict
        return ensembled_predict

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Method make prediction while graph fitting.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        :param input_data: data with features, target and ids to process
        """
        return self.transform(input_data)
