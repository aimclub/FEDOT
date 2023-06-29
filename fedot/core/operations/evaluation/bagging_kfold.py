from abc import ABC
from typing import Optional

from catboost import CatBoostClassifier, CatBoostRegressor
from golem.core.utilities.random import RandomStateHandler
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.bag_ensembles.bag_ensemble import \
    KFoldBaggingClassifier, KFoldBaggingRegressor
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


class KFoldBaggingStrategy(EvaluationStrategy, ABC):
    """ This class defines the certain multi-layer stack ensembling n-repeated k-fold bagging implementation

    Args:
        operation_type: 'str' selected operation as a base model in bagging

        .. details:: possible bagging operations:

            - ``bag_catboost`` -> Bagging for the CatBoost
            - ``bag_lgbm`` -> Bagging for the LightGBM
            - ``bag_xgboost`` -> Bagging for the XGBoost

        params: operation's init and fitting hyperparameters

        .. details:: explanation of params
            - ``model_base`` -
            - ``n_repeats`` -
            - ``k_fold`` -
            - ``fold_fitting_strategy`` -
            - ``n_jobs`` -
            - ``model_base_kwargs`` -

    """

    _operations_by_types = {
        # Classification
        'bag_catboost': CatBoostClassifier,
        'bag_xgboost': XGBClassifier,
        'bag_lgbm': LGBMClassifier,
        'bag_lgbmxt': LGBMClassifier,

        # Regression
        'bag_catboostreg': CatBoostRegressor,
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
                self._bagging_params['model_base'] = self._operations_by_types[operation_type](**self._model_params)
            else:
                self._bagging_params['model_base'] = self._operations_by_types[operation_type]()

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

        with RandomStateHandler():
            operation_implementation.fit(train_data.features, train_data.target, train_data.features_type)

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


class KFoldBaggingClassificationStrategy(KFoldBaggingStrategy):
    # TODO: Avoid duplicate with SklearnBagging,
    #  implement it with optional of bagging_operation param
    """ Classification bagging operation implementation

        Args:
            operation_type: 'str' selected operation as a base model in bagging
            params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = KFoldBaggingClassifier
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['default', 'labels']:
            prediction = trained_operation.predict(predict_data.features)

        elif self.output_mode in ['probs', 'full_probs'] and predict_data.task:
            prediction = trained_operation.predict_proba(predict_data.features)

        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class KFoldBaggingRegressionStrategy(KFoldBaggingStrategy):
    # TODO: Avoid duplicate with SklearnBagging,
    #  implement it with optional of bagging_operation param
    """ Regression bagging operation implementation

        Args:
            operation_type: 'str' selected operation as a base model in bagging
            params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = KFoldBaggingRegressor
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data.features)

        return self._convert_to_output(prediction, predict_data)



