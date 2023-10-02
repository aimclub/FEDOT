from abc import ABC
from typing import Optional

from golem.core.utilities.random import RandomStateHandler
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


class SkLearnBaggingStrategy(EvaluationStrategy, ABC):
    """ This class defines the certain base bagging operation implementation for the sklearn operations
    defined in operation repository

    Args:
        operation_type: 'str' selected operation as a base model in bagging

        .. details:: possible bagging operations:

            - ``bag_dt`` -> Bagging for the Decision Tree
            - ``bag_dtreg`` -> Bagging for the Decision Trees Regressors
            - ``bag_adareg`` -> Bagging for AdaBoosting Regressor

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

    _operations_by_types = {
        # Classification
        'bag_dtreg': DecisionTreeRegressor,
        'bag_adareg': AdaBoostRegressor,

        # Regression
        'bag_dt': DecisionTreeClassifier,
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

        self._model_params = params.get('model_params')
        # TODO: sklearn param base_estimator will change to estimator in future since 1.4

        self._bagging_params = {}

        for param in params.keys():
            if param != 'model_params':
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


class SkLearnBaggingClassificationStrategy(SkLearnBaggingStrategy):
    """ Classification bagging operation implementation

        Args:
            operation_type: 'str' selected operation as a base model in bagging
            params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingClassifier
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['default', 'labels']:
            prediction = trained_operation.predict(predict_data.features)

        elif self.output_mode in ['probs', 'full_probs'] and predict_data.task:
            prediction = trained_operation.predict_proba(predict_data.features)

        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class SkLearnBaggingRegressionStrategy(SkLearnBaggingStrategy):
    """ Regression bagging operation implementation

        Args:
            operation_type: 'str' selected operation as a base model in bagging
            params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingRegressor
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data.features)

        return self._convert_to_output(prediction, predict_data)
