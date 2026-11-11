from typing import Optional
import warnings

import torch
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

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.models.estimator_adapters import (
    ExplainableBoostingClassifierAdapter,
    ExplainableBoostingRegressorAdapter,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.utils import is_multi_output_target
from fedot.utilities.random import ImplementationRandomStateHandler


warnings.filterwarnings('ignore', category=UserWarning)


class SklearnModelStrategy(EvaluationStrategy):
    """TensorData boundary for sklearn-compatible estimators computing on CPU.

    Features and target become NumPy arrays only at the estimator boundary;
    predictions return to the input device without changing the input metadata.
    """

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        features = as_numpy(train_data.features)
        target = self._sklearn_compatible_target(as_numpy(train_data.target))
        implementation = self.operation_impl(**self.params_for_fit.to_dict())
        multi_output = is_multi_output_target(train_data)
        non_multi_models = OperationTypesRepository().suitable_operation(
            task_type=train_data.task.task_type, tags=['non_multi'])
        with warnings.catch_warnings(), ImplementationRandomStateHandler(implementation=implementation):
            warnings.simplefilter('ignore', RuntimeWarning)
            if multi_output and self.operation_type in non_multi_models:
                implementation = self._multi_output_wrapper(implementation)
            implementation.is_multi_target = multi_output
            implementation.fit(features, target)
        return implementation

    @staticmethod
    def _sklearn_compatible_target(target):
        if target is not None and target.ndim == 2 and target.shape[1] == 1:
            return target.reshape(-1)
        return target

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        prediction = trained_operation.predict(as_numpy(predict_data.features))
        return self._output(prediction, predict_data)

    def _output(self, prediction, predict_data: TensorData) -> TensorData:
        prediction = as_torch(prediction, device=predict_data.device)
        return self._replace_predict_in_tensor_data(prediction, predict_data)


class SklearnClassificationStrategy(SklearnModelStrategy):
    _multi_output_wrapper = MultiOutputClassifier
    _operations_by_types = {
        'logit': SklearnLogReg,
        'bernb': SklearnBernoulliNB,
        'multinb': SklearnMultinomialNB,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
        'extra_trees': ExtraTreesClassifier,
        'hist_gb': HistGradientBoostingClassifier,
        'ebm': ExplainableBoostingClassifierAdapter,
    }

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        features = as_numpy(predict_data.features)
        if self.output_mode == 'labels':
            return self._output(trained_operation.predict(features), predict_data)
        if self.output_mode not in ['probs', 'full_probs', 'default']:
            raise ValueError(f'Output mode {self.output_mode!r} is not supported')

        classes = trained_operation.classes_
        prediction = trained_operation.predict_proba(features)
        if isinstance(classes, list):
            # Trees and MultiOutputClassifier expose one probability matrix per target.
            if any(len(target_classes) < 2 for target_classes in classes):
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            prediction = torch.stack([as_torch(p, device=predict_data.device) for p in prediction], dim=1)
        else:
            if len(classes) < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            prediction = as_torch(prediction, device=predict_data.device)
            if trained_operation.is_multi_target:
                # MLPClassifier returns positive probabilities for each binary target.
                prediction = torch.stack((1 - prediction, prediction), dim=-1)
        if prediction.shape[-1] == 2 and self.output_mode != 'full_probs':
            prediction = prediction[..., 1]
        return self._output(prediction, predict_data)


class SklearnRegressionStrategy(SklearnModelStrategy):
    _multi_output_wrapper = MultiOutputRegressor
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
        'hist_gbreg': HistGradientBoostingRegressor,
        'ebmreg': ExplainableBoostingRegressorAdapter,
        'mlpreg': MLPRegressor,
    }


class SklearnClusteringStrategy(SklearnModelStrategy):
    _operations_by_types = {'kmeans': SklearnKmeans}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        if not params:
            self.params_for_fit.update(n_clusters=2)

    def fit(self, train_data: TensorData):
        implementation = self.operation_impl(**self.params_for_fit.to_dict())
        with ImplementationRandomStateHandler(implementation=implementation):
            implementation.fit(as_numpy(train_data.features))
        return implementation
