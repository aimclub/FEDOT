import importlib.util
import platform
import warnings
from typing import Optional, Union

import numpy as np
import torch
from golem.utilities.requirements_notificator import warn_requirement

try:
    import cupy
    import cuml
    from cuml.cluster import KMeans
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
    from cuml.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        MBSGDClassifier,
        MBSGDRegressor,
        Ridge,
    )
    from cuml.naive_bayes import BernoulliNB, MultinomialNB
    from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from cuml.solvers import CD, SGD
    from cuml.svm import SVC
except (ImportError, ModuleNotFoundError):
    warn_requirement('cupy / cuml', 'cuml-cu12')
    cupy = None
    cuml = None

from fedot.core.data.input_data.data import InputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.gpu.rules import (
    adapt_cuml_parameters,
    build_cuml_precision_plan,
    build_cuml_runtime_plan,
    validate_cuml_tensor_operation,
)
from fedot.core.operations.evaluation.model_engine_rules import (
    ModelEngine,
    ModelEngineCapabilities,
    ModelEngineRequest,
    build_model_engine_plan,
    resolve_runtime_platform,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler


class CuMLEvaluationStrategy(EvaluationStrategy):
    """Common native cuML strategy for legacy InputData and TensorData."""

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
        self.engine_plan = build_model_engine_plan(
            ModelEngineRequest(
                supported_engines=(ModelEngine.CUML,),
                preferred_engine=ModelEngine.CUML,
                require_acceleration=True,
            ),
            _detect_model_engine_capabilities(),
        )
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: Union[InputData, TensorData]):
        """Fit a cuML estimator and keep TensorData transfers device-to-device."""
        if isinstance(train_data, TensorData):
            validate_cuml_tensor_operation(self.operation_type)

        precision_plan = build_cuml_precision_plan(self.operation_type)
        features, device = self._features_to_cuml(train_data.features, precision_plan.dtype_name)
        target = self._target_to_cuml(train_data.target, precision_plan.dtype_name, device)
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
    def _features_to_cuml(features, dtype_name: str):
        torch_dtype = getattr(torch, dtype_name)
        cupy_dtype = getattr(cupy, dtype_name)
        if isinstance(features, torch.Tensor):
            device = features.device if features.is_cuda else torch.device('cuda')
            features = features.detach().to(device=device, dtype=torch_dtype).contiguous()
            return cupy.from_dlpack(features), device
        return cupy.asarray(np.asarray(features), dtype=cupy_dtype), None

    @staticmethod
    def _target_to_cuml(target, dtype_name: str, device: Optional[torch.device]):
        if target is None:
            return None
        if len(target.shape) > 1 and target.shape[1] != 1:
            raise ValueError('cuML model engines currently support a single target column')
        torch_dtype = getattr(torch, dtype_name)
        cupy_dtype = getattr(cupy, dtype_name)
        if isinstance(target, torch.Tensor):
            target = target.detach().to(device=device or 'cuda', dtype=torch_dtype).reshape(-1).contiguous()
            return cupy.from_dlpack(target)
        return cupy.asarray(np.asarray(target).reshape(-1), dtype=cupy_dtype)

    def _features_and_runtime(self, predict_data: Union[InputData, TensorData]):
        is_tensor_data = isinstance(predict_data, TensorData)
        device = str(predict_data.features.device) if is_tensor_data else 'cpu'
        runtime_plan = build_cuml_runtime_plan(is_tensor_data, device)
        precision_plan = build_cuml_precision_plan(self.operation_type)
        features, _ = self._features_to_cuml(predict_data.features, precision_plan.dtype_name)
        return features, runtime_plan

    @staticmethod
    def _prediction_to_runtime(prediction, runtime_plan):
        prediction = cupy.ascontiguousarray(prediction)
        if runtime_plan.returns_tensor:
            return torch.from_dlpack(prediction).to(runtime_plan.result_device)
        return cupy.asnumpy(prediction)

    def _convert_cuml_output(self, prediction, predict_data, runtime_plan):
        runtime_prediction = self._prediction_to_runtime(prediction, runtime_plan)
        return self._convert_to_output(runtime_prediction, predict_data)


def _detect_model_engine_capabilities() -> ModelEngineCapabilities:
    installed_engines = {
        engine for engine, module_name in (
            (ModelEngine.TORCH, 'torch'),
            (ModelEngine.SKLEARN, 'sklearn'),
        ) if importlib.util.find_spec(module_name) is not None
    }
    if cuml is not None:
        installed_engines.add(ModelEngine.CUML)
    return ModelEngineCapabilities(
        platform=resolve_runtime_platform(platform.system(), platform.release()),
        cuda_available=torch.cuda.is_available(),
        installed_engines=frozenset(installed_engines),
    )
