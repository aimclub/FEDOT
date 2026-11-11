import platform
from typing import Any, Mapping, Optional
import warnings

from golem.utilities.requirements_notificator import warn_requirement
import torch

from fedot.core.backend.array_conversions import as_cupy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.utilities.random import ImplementationRandomStateHandler


try:
    import cupy
    import cuml
    from cuml.cluster import KMeans
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
    from cuml.linear_model import (
        ElasticNet, Lasso, LinearRegression, LogisticRegression, MBSGDClassifier, MBSGDRegressor, Ridge,
    )
    from cuml.naive_bayes import BernoulliNB, MultinomialNB
    from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from cuml.solvers import CD, SGD
    from cuml.svm import SVC
except (ImportError, ModuleNotFoundError):
    warn_requirement('cupy / cuml', 'cuml-cu12')
    cupy = None
    cuml = None


class CuMLModelStrategy(EvaluationStrategy):
    """Common native cuML strategy for TensorData."""

    if cuml is not None:
        _operations_by_types = {
            'ridge': Ridge,
            'lasso': Lasso,
            'logit': LogisticRegression,
            'linear': LinearRegression,
            'rf': RandomForestClassifier,
            'rfr': RandomForestRegressor,
            'svc': SVC,
            'knn': KNeighborsClassifier,
            'knnreg': KNeighborsRegressor,
            'sgd': SGD,
            'multinb': MultinomialNB,
            'bernb': BernoulliNB,
            'elasticnet': ElasticNet,
            'minibatchsgd': MBSGDClassifier,
            'mbsgdcregr': MBSGDRegressor,
            'cd': CD,
            'kmeans': KMeans,
        }
    else:
        _operations_by_types = {}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        if isinstance(params, dict):
            params = OperationParameters(**params)
        super().__init__(operation_type, params)
        _validate_cuml_environment()
        self.dtype = torch.float64 if operation_type == 'logit' else torch.float32
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: TensorData):
        """Fit a cuML estimator and keep TensorData transfers device-to-device."""
        validate_cuml_tensor_operation(self.operation_type)

        features, device = self._features_to_cuml(train_data.features, self.dtype)
        target = self._target_to_cuml(train_data.target, self.dtype, device)
        parameters = adapt_cuml_parameters(self.operation_type, self.params_for_fit.to_dict())
        operation_implementation = self.operation_impl(**parameters)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            with ImplementationRandomStateHandler(implementation=operation_implementation):
                if self.operation_type == 'kmeans':
                    operation_implementation.fit(features)
                else:
                    operation_implementation.fit(features, target)
        return operation_implementation

    @staticmethod
    def _features_to_cuml(features: torch.Tensor, dtype: torch.dtype):
        device = features.device if features.is_cuda else torch.device('cuda')
        features = as_torch(features, device=device, dtype=dtype, contiguous=True)
        return as_cupy(features), device

    @staticmethod
    def _target_to_cuml(target: Optional[torch.Tensor], dtype: torch.dtype, device: torch.device):
        if target is None:
            return None
        if len(target.shape) > 1 and target.shape[1] != 1:
            raise ValueError('cuML model engines currently support a single target column')
        target = as_torch(target.reshape(-1), device=device, dtype=dtype, contiguous=True)
        return as_cupy(target)

    def _convert_cuml_output(self, prediction, predict_data: TensorData) -> TensorData:
        runtime_prediction = as_torch(prediction, device=predict_data.device, contiguous=True)
        return self._replace_predict_in_tensor_data(runtime_prediction, predict_data)


def _validate_cuml_environment():
    if cuml is None:
        raise RuntimeError('cuML is not installed')
    release = platform.release().lower()
    if platform.system().lower() != 'linux' or 'microsoft' in release or 'wsl' in release:
        raise RuntimeError('cuML is enabled only on native Linux')
    if not torch.cuda.is_available():
        raise RuntimeError('cuML requires an available CUDA device')


def adapt_cuml_parameters(operation_type: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate shared FEDOT/sklearn parameters to the native cuML API."""
    adapted = dict(parameters)
    adapted.pop('n_jobs', None)
    if operation_type in {'rf', 'rfr'}:
        if 'criterion' in adapted:
            adapted['split_criterion'] = adapted.pop('criterion')
        adapted.setdefault('n_bins', 256)
    if operation_type == 'logit':
        adapted.setdefault('max_iter', 10000)
        adapted.setdefault('tol', 1e-7)
        adapted.setdefault('linesearch_max_iter', 200)
    if operation_type == 'svc':
        adapted['probability'] = True
    if operation_type == 'kmeans':
        adapted.setdefault('n_clusters', 2)
    adapted['output_type'] = 'cupy'
    return adapted


def validate_cuml_tensor_operation(operation_type: str):
    """Reject the historical solver alias without a model contract."""
    if operation_type == 'sgd':
        raise ValueError(
            f'cuML operation {operation_type!r} is a low-level solver and has no TensorData model contract'
        )


class CuMLClassificationStrategy(CuMLModelStrategy):
    """Classification strategy for TensorData."""

    def predict(
        self,
        trained_operation,
        predict_data: TensorData,
    ) -> TensorData:
        features, _ = self._features_to_cuml(predict_data.features, self.dtype)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default', False]:
            if not hasattr(trained_operation, 'predict_proba'):
                if self.output_mode in ['probs', 'full_probs']:
                    raise ValueError(f'cuML {self.operation_type!r} does not provide class probabilities')
                prediction = trained_operation.predict(features)
            else:
                prediction = trained_operation.predict_proba(features)
                n_classes = prediction.shape[1]
                if n_classes < 2:
                    raise ValueError('Data set contains only 1 target class. Please reformat your data.')
                if n_classes == 2 and self.output_mode != 'full_probs':
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output mode {self.output_mode!r} is not supported')
        return self._convert_cuml_output(prediction, predict_data)


class CuMLRegressionStrategy(CuMLModelStrategy):
    """Regression strategy for TensorData."""

    def predict(
        self,
        trained_operation,
        predict_data: TensorData,
    ) -> TensorData:
        features, _ = self._features_to_cuml(predict_data.features, self.dtype)
        prediction = trained_operation.predict(features)
        return self._convert_cuml_output(prediction, predict_data)


class CuMLClusteringStrategy(CuMLModelStrategy):
    """K-means strategy for TensorData."""

    def predict(
        self,
        trained_operation,
        predict_data: TensorData,
    ) -> TensorData:
        features, _ = self._features_to_cuml(predict_data.features, self.dtype)
        prediction = trained_operation.predict(features)
        return self._convert_cuml_output(prediction, predict_data)
