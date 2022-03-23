import warnings
from abc import abstractmethod
from typing import Optional

from catboost import CatBoostClassifier, CatBoostRegressor
from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log, default_log
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_operation_type_from_id
from fedot.core.repository.tasks import TaskTypesEnum
from lightgbm import LGBMClassifier, LGBMRegressor
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

warnings.filterwarnings("ignore", category=UserWarning)


class EvaluationStrategy:
    """
    Base class to define the evaluation strategy of Operation object:
    the certain sklearn or any other operation with fit/predict methods.
    :param operation_type: str type of the operation defined in operation repository
    :param dict params: hyperparameters to fit the operation with
    :param Log log: Log object to record messages
    """

    def __init__(self, operation_type: str, params: Optional[dict] = None,
                 log=None):
        self.params_for_fit = params
        self.operation_id = operation_type

        self.output_mode = False

        if not log:
            self.log: Log = default_log(__name__)
        else:
            self.log: Log = log

    @property
    def operation_type(self):
        return get_operation_type_from_id(self.operation_id)

    @abstractmethod
    def fit(self, train_data: InputData):
        """
        Main method to train the operation with the data provided
        :param InputData train_data: data used for operation training
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        """
        Main method to predict the target data.
        :param trained_operation: trained operation object
        :param InputData predict_data: data to predict
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    @abstractmethod
    def _convert_to_operation(self, operation_type: str):
        raise NotImplementedError()

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    @staticmethod
    def _convert_to_output(prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        """ Method convert prediction into OutputData if it is not this type yet

        :param prediction: output from model implementation
        :param predict_data: InputData used for prediction
        :param output_data_type: DataTypesEnum for output

        :return : prediction as OutputData
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
    """
    This class defines the certain operation implementation for the sklearn operations
    defined in operation repository
    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the operation with
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

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.operation_id = operation_type
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained Sklearn operation
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models, _ = models_repo.suitable_operation(task_type=current_task,
                                                             tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models

        # Multi-output task or not
        is_multi_target = is_multi_output_task(train_data)
        if is_model_not_support_multi and is_multi_target:
            # Manually wrap the regressor into multi-output model
            operation_implementation = convert_to_multivariate_model(operation_implementation,
                                                                     train_data)
        else:
            operation_implementation.fit(train_data.features, train_data.target)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.
        :param trained_operation: operation object
        :param predict_data: data to predict
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return OutputData: passed data with new predicted target
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
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return prediction


def convert_to_multivariate_model(sklearn_model, train_data: InputData):
    """
    The function returns an iterator for multiple target for those models for
    which such a function is not initially provided

    :param sklearn_model: Sklearn model to train
    :param train_data: data used for model training
    :return : wrapped Sklearn model
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
