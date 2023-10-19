from abc import ABC
from typing import Optional

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.bag_ensembles.bag_ensemble import \
    KFoldBaggingClassifier, KFoldBaggingRegressor
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import \
    FedotCatBoostClassificationImplementation, FedotCatBoostRegressionImplementation
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params
from fedot.utilities.random import ImplementationRandomStateHandler


class BaseBaggingStrategy(EvaluationStrategy, ABC):
    """ This class defines base bagging operations implementation """

    _operations_by_types = {
        # Classification
        # Sklearn Bagging Strategy
        'bag_dtreg': DecisionTreeRegressor,
        'bag_adareg': AdaBoostRegressor,
        # K-fold Bagging Strategy
        'bag_catboost': FedotCatBoostClassificationImplementation,
        'bag_xgboost': XGBClassifier,
        'bag_lgbm': LGBMClassifier,
        'bag_lgbmxt': LGBMClassifier,

        # Regression
        # Sklearn Bagging Strategy
        'bag_dt': DecisionTreeClassifier,
        # K-fold Bagging Strategy
        'bag_catboostreg': FedotCatBoostRegressionImplementation,
        'bag_xgboostreg': XGBRegressor,
        'bag_lgbmreg': LGBMRegressor,
        'bag_lgbmxtreg': LGBMRegressor,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = None
        self.bagging_operation = None

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types.keys():
            if self._model_params:
                self._bagging_params['estimator'] = self._operations_by_types[operation_type](**self._model_params)
            else:
                self._bagging_params['estimator'] = self._operations_by_types[operation_type]()

            return self.bagging_operation(**self._bagging_params)

        else:
            raise ValueError(f'Impossible to create bagging operation for {operation_type}')

    def _set_operation_params(self, operation_type, params):
        if params is None:
            params = get_default_params(operation_type)

        elif isinstance(params, dict):
            params = OperationParameters.from_operation_type(operation_type, **params)

        elif isinstance(params, OperationParameters):
            # Getting models params after applying mutation
            if params.get('model_params'):
                params = OperationParameters.from_operation_type(operation_type, **(params.to_dict()))

        self._model_params = params.get('model_base_kwargs')
        self._bagging_params = {}

        for param in params.keys():
            if param != 'model_base_kwargs':
                self._bagging_params.update({param: params.get(param)})

        return params

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def fit(self, train_data: InputData):
        """ Method to train chosen operation with provided data

        Args:
            train_data: data used for operation training

        Returns:
            trained bagging model
        """
        operation_implementation = self.operation_impl

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data.features, train_data.target)

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData):
        """ This method used for prediction of the target data

        Args:
            trained_operation: operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        NotImplementedError()


class BaseBaggingClassification(BaseBaggingStrategy):
    """ This class defines general methods for classification problem. """
    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['default', 'labels']:
            prediction = trained_operation.predict(predict_data.features)

        elif self.output_mode in ['probs', 'full_probs'] and predict_data.task:
            prediction = trained_operation.predict_proba(predict_data.features)

        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class BaseBaggingRegression(BaseBaggingStrategy):
    """ This class defines general methods for Regression problem. """
    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data.features)

        return self._convert_to_output(prediction, predict_data)


class SkLearnBaggingClassificationStrategy(BaseBaggingStrategy):
    """ Bagging with the SklearnBaggingClassifier

    Args:
        operation_type: 'str' selected operation as a base model (estimator) in bagging
        .. details:: possible bagging operations for classification:
            - ``bag_dt`` -> Bagging for the Decision Tree
            - ``bag_dtreg`` -> Bagging for the Decision Trees Regressors

        params: operation's init and fitting hyperparameters
        .. details:: explanation of params
            - ``n_estimators`` - the number of base estimators in bagging ensemble
            - ``bootstrap`` - whether samples are drawn with replacement. If False,
                              sampling without replacement is performed.
            - ``oob_score`` - whether to use out-of-bag samples to estimate the generalization error.
                              Only available if bootstrap=True.
            - ``max_samples`` - the number of samples to draw from X to train each base estimator
            - ``max_features`` - the number of features to draw from X to train each base estimator
            - ``n_jobs`` - the number of jobs to run in parallel
            - ``model_params`` - model's fitting params
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingClassifier


class SkLearnBaggingRegressionStrategy(BaseBaggingStrategy):
    """ Bagging with the SklearnBaggingRegressor

    Args:
        operation_type: 'str' selected operation as a base model (estimator) in bagging
        .. details:: possible bagging operations:
            - ``bag_adareg`` -> Bagging for the AdaBoosting Regressor

        params: operation's init and fitting hyperparameters
        .. details:: explanation of params
            - ``n_estimators`` - the number of base estimators in bagging ensemble
            - ``bootstrap`` - whether samples are drawn with replacement. If False,
                              sampling without replacement is performed.
            - ``oob_score`` - whether to use out-of-bag samples to estimate the generalization error.
                              Only available if bootstrap=True.
            - ``max_samples`` - the number of samples to draw from X to train each base estimator
            - ``max_features`` - the number of features to draw from X to train each base estimator
            - ``n_jobs`` - the number of jobs to run in parallel
            - ``model_params`` - model's fitting params
    """
    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingRegressor


class KFoldBaggingClassificationStrategy(BaseBaggingStrategy):
    """ Bagging with the KFoldBaggingClassifier (K-fold n-repeated bagging)

    Args:
        operation_type: 'str' selected operation as a base model in bagging
        .. details:: possible bagging operations:
            - ``bag_catboost`` -> Bagging for the CatBoost

        params: operation's init and fitting hyperparameters
        .. details:: explanation of params
            - ``k_fold`` - the number of data splits and base estimators in bagging ensembles
            - ``n_repeats`` - the number of fold fitting repeats per each estimator
            - ``fold_fitting_strategy`` - the fitting strategy
            - ``n_jobs`` - the number of jobs to run in parallel
            - ``model_params`` - model's fitting params
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.bagging_operation = KFoldBaggingClassifier


class KFoldBaggingRegressionStrategy(BaseBaggingStrategy):
    """ Bagging with the KFoldBaggingRegressor (K-fold n-repeated bagging)

    Args:
        operation_type: 'str' selected operation as a base model in bagging
        .. details:: possible bagging operations:
            - ``bag_catboostreg`` -> Bagging for the CatBoost

        params: operation's init and fitting hyperparameters
        .. details:: explanation of params
            - ``k_fold`` - the number of data splits and base estimators in bagging ensembles
            - ``n_repeats`` - the number of fold fitting repeats per each estimator
            - ``fold_fitting_strategy`` - the fitting strategy
            - ``n_jobs`` - the number of jobs to run in parallel
            - ``model_params`` - model's fitting params
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.bagging_operation = KFoldBaggingRegressor
