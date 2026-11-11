from abc import abstractmethod
from typing import Optional
import warnings

import numpy as np
from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LinearRegression as SklearnLinReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD,
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import torch

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utils import is_multi_output_model, is_multi_output_target
from fedot.utilities.custom_errors import AbstractMethodNotImplementError
from fedot.utilities.random import ImplementationRandomStateHandler


warnings.filterwarnings('ignore', category=UserWarning)


class SklearnModelStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the sklearn operations
    defined in operation repository

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``adareg``-> AdaBoostRegressor
                - ``gbr``-> GradientBoostingRegressor
                - ``dtreg``-> DecisionTreeRegressor
                - ``treg``-> ExtraTreesRegressor
                - ``rfr``-> RandomForestRegressor
                - ``linear``-> SklearnLinReg
                - ``ridge``-> SklearnRidgeReg
                - ``lasso``-> SklearnLassoReg
                - ``svr``-> SklearnSVR
                - ``sgdr``-> SklearnSGD
                - ``logit``-> SklearnLogReg
                - ``bernb``-> SklearnBernoulliNB
                - ``multinb``-> SklearnMultinomialNB
                - ``dt``-> DecisionTreeClassifier
                - ``rf``-> RandomForestClassifier
                - ``mlp``-> MLPClassifier

        params: hyperparameters to fit the operation with
    """

    _operations_by_types = {
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rfr': RandomForestRegressor,
        'linear': SklearnLinReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'svr': SklearnSVR,
        'sgdr': SklearnSGD,

        'logit': SklearnLogReg,
        'bernb': SklearnBernoulliNB,
        'multinb': SklearnMultinomialNB,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        operation_implementation = self.operation_impl(
            **self.params_for_fit.to_dict())

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models = models_repo.suitable_operation(task_type=current_task,
                                                          tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_target(train_data)
        self.operation_impl.is_multi_target = is_multi_target

        with ImplementationRandomStateHandler(implementation=operation_implementation):
            if is_model_not_support_multi and is_multi_target:
                # Manually wrap the regressor into multi-output model
                operation_implementation = convert_to_multivariate_model(
                    operation_implementation, train_data)
            else:
                target = self._sklearn_compatible_target(train_data.target)
                operation_implementation.fit(
                    train_data.features, target)

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

    def _find_operation_by_impl(self, impl):
        for operation, operation_impl in self._operations_by_types.items():
            if operation_impl == impl:
                return operation

    @staticmethod
    def _sklearn_compatible_target(target):
        if target is not None and len(target.shape) == 2 and target.shape[1] == 1:
            return target.reshape(-1)
        return target

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _sklearn_compatible_prediction(self, trained_operation, features):
        # Check if target is multilabel (has 2 or more columns)
        if is_multi_output_model(self.operation_impl):
            n_classes = len(trained_operation.classes_[0])
        else:
            n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise ValueError(
                    'Data set contain only 1 target class. Please reformat your data.')
            elif n_classes == 2 and self.output_mode != 'full_probs':
                if is_multi_output_model(self.operation_impl):
                    prediction = np.stack([pred[:, 1]
                                          for pred in prediction]).T
                else:
                    prediction = prediction[:, 1]
        else:
            raise ValueError(
                f'Output model {self.output_mode} is not supported')

        return prediction


def convert_to_multivariate_model(sklearn_model, train_data: InputData):
    """The function returns an iterator for multiple target for those models for
    which such a function is not initially provided

    Args:
        sklearn_model: :obj:`Sklearn model` to train
        train_data: data used for model training
    Returns:
        wrapped :obj:`Sklearn model`
    """

    if train_data.task.task_type == TaskTypesEnum.classification:
        multiout_func = MultiOutputClassifier
    elif train_data.task.task_type in [TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting]:
        multiout_func = MultiOutputRegressor
    else:
        raise ValueError(
            f"For task type '{train_data.task.task_type}' MultiOutput wrapper is not supported")

    # Apply MultiOutput
    sklearn_model = multiout_func(sklearn_model)
    sklearn_model.fit(train_data.features, train_data.target)
    return sklearn_model


class SklearnClassificationStrategy(SklearnModelStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for classification task for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """

        prediction = self._sklearn_compatible_prediction(
            trained_operation=trained_operation,
            features=predict_data.features
        )

        converted = self._convert_to_output(prediction, predict_data)
        return converted


class SklearnRegressionStrategy(SklearnModelStrategy):
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for regression task for predict stage
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return:
        """
        prediction = trained_operation.predict(predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)

        return converted


class SklearnClusteringStrategy(SklearnModelStrategy):
    _operations_by_types = {
        'kmeans': SklearnKmeans
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        if not params:
            self.params_for_fit.update(n_clusters=2)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(
            **self.params_for_fit.to_dict())
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data.features)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for clustering task
        :param trained_operation: operation object
        :param predict_data: data used for prediction
        """
        prediction = trained_operation.predict(predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
