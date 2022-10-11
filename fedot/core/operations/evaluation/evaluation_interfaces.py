import warnings
from abc import abstractmethod
from typing import Optional

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans as SklearnKmeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import (
    Lasso as SklearnLassoReg,
    LinearRegression as SklearnLinReg,
    LogisticRegression as SklearnLogReg,
    Ridge as SklearnRidgeReg,
    SGDRegressor as SklearnSGD
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operation_type_from_id
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utilities.random import RandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    """Base class to define the evaluation strategy of Operation object:
    the certain sklearn or any other operation with fit/predict methods.

    Args:
        operation_type: ``str`` of the operation defined in operation repository
        params: hyperparameters to fit the operation with
    """

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.params_for_fit = params or OperationParameters()
        self.operation_id = operation_type

        self.output_mode = False

        self.log = default_log(self)

    @property
    def operation_type(self):
        return get_operation_type_from_id(self.operation_id)

    @abstractmethod
    def fit(self, train_data: InputData):
        """Main method to train the operation with the data provided

        Args:
            train_data: data used for operation training

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for predict stage.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        raise NotImplementedError()

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for fit stage.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict
        Returns:
            passed data with new predicted target
        """
        return self.predict(trained_operation, predict_data)

    @abstractmethod
    def _convert_to_operation(self, operation_type: str):
        raise NotImplementedError()

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    @staticmethod
    def _convert_to_output(prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        """Method convert prediction into :obj:`OutputData` if it is not this type yet

        Args:
            prediction: output from model implementation
            predict_data: :obj:`InputData` used for prediction
            output_data_type: :obj:`DataTypesEnum` for output

        Returns: prediction as :obj:`OutputData`
        """

        if type(prediction) is not OutputData:
            # Wrap prediction as OutputData
            converted = OutputData(idx=predict_data.idx,
                                   features=predict_data.features,
                                   predict=prediction,
                                   task=predict_data.task,
                                   target=predict_data.target,
                                   data_type=output_data_type,
                                   supplementary_data=predict_data.supplementary_data)
        else:
            converted = prediction

        return converted


class SkLearnEvaluationStrategy(EvaluationStrategy):
    """This class defines the certain operation implementation for the sklearn operations
    defined in operation repository

    Args:
        operation_type: ``str`` of the operation defined in operation or
            data operation repositories

            .. details:: possible operations:

                - ``xgbreg``-> XGBRegressor
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
                - ``lgbmreg``-> LGBMRegressor
                - ``catboostreg``-> CatBoostRegressor
                - ``xgboost``-> XGBClassifier
                - ``logit``-> SklearnLogReg
                - ``bernb``-> SklearnBernoulliNB
                - ``multinb``-> SklearnMultinomialNB
                - ``dt``-> DecisionTreeClassifier
                - ``rf``-> RandomForestClassifier
                - ``mlp``-> MLPClassifier
                - ``lgbm``-> LGBMClassifier
                - ``catboost``-> CatBoostClassifier
                - ``kmeans``-> SklearnKmeans

        params: hyperparameters to fit the operation with
    """

    __operations_by_types = {
        'xgbreg': XGBRegressor,
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
        'lgbmreg': LGBMRegressor,
        'catboostreg': CatBoostRegressor,

        'xgboost': XGBClassifier,
        'logit': SklearnLogReg,
        'bernb': SklearnBernoulliNB,
        'multinb': SklearnMultinomialNB,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
        'lgbm': LGBMClassifier,
        'catboost': CatBoostClassifier,

        'kmeans': SklearnKmeans,
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

        operation_implementation = self.operation_impl(**self.params_for_fit.to_dict())

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models = models_repo.suitable_operation(task_type=current_task,
                                                          tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_task(train_data)
        if is_model_not_support_multi and is_multi_target:
            # Manually wrap the regressor into multi-output model
            operation_implementation = convert_to_multivariate_model(operation_implementation,
                                                                     train_data)
        else:
            with RandomStateHandler():
                operation_implementation.fit(train_data.features, train_data.target)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """This method used for prediction of the target data

        Args:
            trained_operation: operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        raise NotImplementedError()

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain SKlearn strategy for {operation_type}')

    def _find_operation_by_impl(self, impl):
        for operation, operation_impl in self.__operations_by_types.items():
            if operation_impl == impl:
                return operation

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _sklearn_compatible_prediction(self, trained_operation, features):
        is_multi_output_target = isinstance(trained_operation.classes_, list)
        # Check if target is multilabel (has 2 or more columns)
        if is_multi_output_target:
            n_classes = len(trained_operation.classes_[0])
        else:
            n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                if is_multi_output_target:
                    prediction = np.stack([pred[:, 1] for pred in prediction]).T
                else:
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

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
        raise ValueError(f"For task type '{train_data.task.task_type}' MultiOutput wrapper is not supported")

    # Apply MultiOutput
    sklearn_model = multiout_func(sklearn_model)
    sklearn_model.fit(train_data.features, train_data.target)
    return sklearn_model


def is_multi_output_task(train_data):
    target_shape = train_data.target.shape
    is_multi_target = len(target_shape) > 1 and target_shape[1] > 1
    return is_multi_target
