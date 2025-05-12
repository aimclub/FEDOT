from typing import Optional

from golem.core.log import default_log
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


class BaggingImplementation(ModelImplementation):
    """Base class for bagging operations"""
    __operation_params = ['n_jobs']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.log = default_log('Bagging')
        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.seed = 42
        self.model = None

    def fit(self, input_data: InputData):
        """Fit the bagging model. Decision Tree estimator set as default.
        Args:
            input_data: Input data features.
        """
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make labels predictions using the bagging model.
        Args:
            input_data: Input data features.
        """
        labels = self.model.predict(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Make probabilities predictions using the bagging model.
        Args:
            input_data: Input data features.
        """
        if input_data.task == TaskTypesEnum.regression or input_data.task == TaskTypesEnum.ts_forecasting:
            raise ValueError('This method does not support regression or time series forecasting tasks')

        probs = self.model.predict_proba(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data


class CatBoostBaggingClassification(BaggingImplementation):
    """CatBoost Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  CatBoostClassifier(**self.model_params)
        self.model = BaggingClassifier(estimator=est)


class CatBoostBaggingRegression(BaggingImplementation):
    """CatBoost Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  CatBoostRegressor(**self.model_params)
        self.model = BaggingRegressor(estimator=est)


class XGBoostBaggingClassification(BaggingImplementation):
    """XGBoost implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  XGBClassifier(**self.model_params)
        self.model = BaggingClassifier(estimator=est)


class XGBoostBaggingRegression(BaggingImplementation):
    """XGBoost Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  XGBRegressor(**self.model_params)
        self.model = BaggingRegressor(estimator=est)


class LGBMBaggingClassification(BaggingImplementation):
    """LGBM Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  LGBMClassifier(**self.model_params)
        self.model = BaggingClassifier(estimator=est)


class LGBMBaggingRegression(BaggingImplementation):
    """LGBM Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  LGBMRegressor(**self.model_params)
        self.model = BaggingRegressor(estimator=est)


class RFBaggingClassification(BaggingImplementation):
    """Random Forest Bagging implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  RandomForestClassifier(**self.model_params)
        self.model = BaggingClassifier(estimator=est)


class RFBaggingRegression(BaggingImplementation):
    """Random Forest Bagging implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        est =  RandomForestRegressor(**self.model_params)
        self.model = BaggingRegressor(estimator=est)
