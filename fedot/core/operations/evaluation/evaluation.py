import warnings

from abc import abstractmethod
from typing import Optional, Type

from sklearn.cluster import KMeans as SklearnKmeans
from fedot.core.operations.evaluation.operation_realisations.models.\
    discriminant_analysis import LDAModel, QDAModel
from sklearn.ensemble import (AdaBoostRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import (Lasso as SklearnLassoReg,
                                  LinearRegression as SklearnLinReg,
                                  LogisticRegression as SklearnLogReg,
                                  Ridge as SklearnRidgeReg,
                                  SGDRegressor as SklearnSGD)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB, MultinomialNB as SklearnMultinomialNB
from sklearn.neighbors import (KNeighborsClassifier as SklearnKNN,
                               KNeighborsRegressor as SklearnKNNReg)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR as SklearnSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log, default_log

from fedot.core.operations.evaluation.operation_realisations.models.svc import CustomSVC

from fedot.core.operations.evaluation.operation_realisations.\
    data_operations.sklearn_filters import LinearRegRANSAC, NonLinearRegRANSAC

from fedot.core.operations.evaluation.operation_realisations.\
    data_operations.sklearn_selectors import LinearRegFS, NonLinearRegFS, LinearClassFS, NonLinearClassFS

from fedot.core.operations.evaluation.operation_realisations.data_operations.\
    sklearn_transformations import PCAOperation, PolyFeaturesOperation, OneHotEncodingOperation, \
    ScalingOperation, NormalizationOperation, KernelPCAOperation, ImputationOperation

from fedot.core.operations.evaluation.operation_realisations.models.ts_models import ARIMAModel, AutoRegModel

from fedot.core.operations.evaluation.operation_realisations.data_operations.ts_transformations \
    import LaggedTransformation, TsSmoothing, ExogDataTransformation

from fedot.core.operations.tuning.tuners import Tuner, SklearnCustomRandomTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

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
        self.operation_type = operation_type

        self.output_mode = False

        if not log:
            self.log: Log = default_log(__name__)
        else:
            self.log: Log = log

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
                is_fit_chain_stage: bool) -> OutputData:
        """
        Main method to predict the target data.

        :param trained_operation: trained operation object
        :param InputData predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    @property
    def implementation_info(self) -> str:
        return 'No description'


class SkLearnEvaluationStrategy(EvaluationStrategy):
    """
    This class defines the certain operation implementation for the sklearn operations
    defined in operation repository

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the operation with
    """
    __operations_by_types = {
        'xgboost': XGBClassifier,
        'xgbreg': XGBRegressor,
        'adareg': AdaBoostRegressor,
        'gbr': GradientBoostingRegressor,
        'knn': SklearnKNN,
        'knnreg': SklearnKNNReg,
        'dt': DecisionTreeClassifier,
        'dtreg': DecisionTreeRegressor,
        'treg': ExtraTreesRegressor,
        'rf': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'mlp': MLPClassifier,
        'lda': LDAModel,
        'qda': QDAModel,
        'linear': SklearnLinReg,
        'logit': SklearnLogReg,
        'ridge': SklearnRidgeReg,
        'lasso': SklearnLassoReg,
        'kmeans': SklearnKmeans,
        'svc': CustomSVC,
        'svr': SklearnSVR,
        'sgdr': SklearnSGD,
        'bernb': SklearnBernoulliNB,
        'multinb': SklearnMultinomialNB
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self._sklearn_operation_impl = self._convert_to_sklearn(operation_type)
        self._tune_strategy: Type[Tuner] = SklearnCustomRandomTuner
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return: trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            sklearn_operation = self._sklearn_operation_impl(**self.params_for_fit)
        else:
            sklearn_operation = self._sklearn_operation_impl()

        sklearn_operation.fit(train_data.features, train_data.target)
        return sklearn_operation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    def _convert_to_sklearn(self, operation_type: str):
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
        return str(self._convert_to_sklearn(self.operation_type))


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for classification task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data.features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data.features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=prediction,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)
        return converted


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for regression task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """

        prediction = trained_operation.predict(predict_data.features)
        # Wrap prediction as features for next level
        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=prediction,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)

        return converted


class SkLearnClusteringStrategy(SkLearnEvaluationStrategy):
    def fit(self, train_data: InputData):
        """
        Fit method for clustering task

        :param train_data: data used for model training
        :return:
        """
        sklearn_model = self._sklearn_operation_impl(n_clusters=2)
        sklearn_model = sklearn_model.fit(train_data.features)
        return sklearn_model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        Predict method for clustering task
        :param trained_operation: operation object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """
        prediction = trained_operation.predict(predict_data.features)
        return prediction


class SkLearnPreprocessingStrategy(EvaluationStrategy):
    __operations_by_types = {
        'scaling': ScalingOperation,
        'normalization': NormalizationOperation,
        'simple_imputation': ImputationOperation,
        'pca': PCAOperation,
        'kernel_pca': KernelPCAOperation,
        'poly_features': PolyFeaturesOperation,
        'one_hot_encoding': OneHotEncodingOperation,
        'ransac_lin_reg': LinearRegRANSAC,
        'ransac_non_lin_reg': NonLinearRegRANSAC,
        'rfe_lin_reg': LinearRegFS,
        'rfe_non_lin_reg': NonLinearRegFS,
        'rfe_lin_class': LinearClassFS,
        'rfe_non_lin_class': NonLinearClassFS
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self._sklearn_operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return:
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            sklearn_operation = self._sklearn_operation_impl(
                **self.params_for_fit)
        else:
            sklearn_operation = self._sklearn_operation_impl()

        sklearn_operation.fit(train_data)
        return sklearn_operation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Transform method for preprocessing task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """
        # Prediction here is already OutputData type object
        prediction = trained_operation.transform(predict_data,
                                                 is_fit_chain_stage)
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain SkLearnPreprocessing strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


class TsTransformingStrategy(EvaluationStrategy):
    """
    This class defines the certain data operation implementation for time series
    forecasting

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'lagged': LaggedTransformation,
        'smoothing': TsSmoothing,
        'exog': ExogDataTransformation}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation = self._convert_to_operation(operation_type)
        self.params_for_fit = params
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return: trained operation (if it is needed for applying)
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            transformation_operation = self.operation(**self.params_for_fit)
        else:
            transformation_operation = self.operation()

        transformation_operation.fit(train_data)
        return transformation_operation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data,
                                                 is_fit_chain_stage)
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain TsTransforming strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


class TsForecastingStrategy(EvaluationStrategy):
    """
    This class defines the certain classical models implementation for time
    series forecasting (e.g. AR, ARIMA)

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'arima': ARIMAModel,
        'ar': AutoRegModel}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation = self._convert_to_operation(operation_type)
        self.params_for_fit = params
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return: trained model
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            model = self.operation(**self.params_for_fit)
        else:
            model = self.operation()

        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.predict(predict_data,
                                               is_fit_chain_stage)
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain TsForecasting strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


def convert_to_multivariate_model_manually(sklearn_model, train_data: InputData):
    """
    The function returns an iterator for multiple target for those models for
    which such a function is not initially provided

    :param sklearn_model: Sklearn model to train
    :param train_data: data used for model training
    :return : wrapped Sklearn model
    """

    if train_data.task.task_type == TaskTypesEnum.classification:
        multiout_func = MultiOutputClassifier
    elif train_data.task.task_type in \
            [TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting]:
        multiout_func = MultiOutputRegressor
    else:
        return None

    # apply MultiOutput
    sklearn_model = multiout_func(sklearn_model)
    sklearn_model.fit(train_data.features, train_data.target)
    return sklearn_model
