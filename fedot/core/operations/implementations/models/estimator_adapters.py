from typing import Callable, Optional

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.utils import is_multi_output_target
from fedot.core.operations.implementations.base import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class SklearnModelAdapter(ModelImplementation):
    """Common CPU-estimator boundary; all public predictions are Torch tensors."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, train_data: TensorData):
        self.is_multi_target = is_multi_output_target(train_data)
        self.model.fit(as_numpy(train_data.features), as_numpy(train_data.target))
        return self.model

    def predict(self, input_data: TensorData) -> torch.Tensor:
        prediction = self.model.predict(as_numpy(input_data.features))
        return as_torch(prediction, device=input_data.device)

    def predict_proba(self, input_data: TensorData) -> torch.Tensor:
        prediction = self.model.predict_proba(as_numpy(input_data.features))
        if isinstance(prediction, list):
            # Native KNN multioutput: one matrix per target, samples axis first.
            return torch.stack([as_torch(p, device=input_data.device) for p in prediction], dim=1)
        return as_torch(prediction, device=input_data.device)


class KNeighborsImplementation(SklearnModelAdapter):
    def check_and_correct_k_value(self, input_data: TensorData, model_impl: Callable):
        """Keep the historical correction for a training set smaller than k."""
        n_neighbors = self.model.n_neighbors
        if n_neighbors > len(input_data.features):
            new_k_value = correct_n_neighbors(n_neighbors, len(input_data.features))
            self.params.update(n_neighbors=new_k_value)
            self.model = model_impl(**self.params.to_dict())
            self.log.info(f'n_neighbors of K-nn model was changed from {n_neighbors} to {new_k_value}')


class KnnClassificationAdapter(KNeighborsImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        params = round_n_neighbors(self.params.to_dict())
        self.model = KNeighborsClassifier(**params)
        self.classes = None

    def fit(self, train_data: TensorData):
        self.classes = torch.unique(train_data.target)
        self.check_and_correct_k_value(train_data, KNeighborsClassifier)
        return super().fit(train_data)

    @property
    def classes_(self):
        return self.classes


class KnnRegressionAdapter(KNeighborsImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        params = round_n_neighbors(self.params.to_dict())
        self.model = KNeighborsRegressor(**params)

    def fit(self, train_data: TensorData):
        self.check_and_correct_k_value(train_data, KNeighborsRegressor)
        return super().fit(train_data)


def correct_n_neighbors(n_neighbors: int, n_samples: int) -> int:
    value = round(n_neighbors)
    value = 1 if value == 0 else value
    return max(1, round(n_samples / 2)) if value > n_samples else value


def round_n_neighbors(params: dict) -> dict:
    """Round n_neighbors in place; preserve the minimum for values rounded to zero."""
    if 'n_neighbors' in params:
        n_neighbors = round(params['n_neighbors'])
        if n_neighbors == 0:
            n_neighbors = 1
        params['n_neighbors'] = n_neighbors
    return params


class FedotSVCImplementation(SklearnModelAdapter):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        if not self.params:
            default_params = {'kernel': 'linear',
                              'probability': True, 'class_weight': 'balanced'}
            self.params.update(**default_params)
        self.inner_model = SVC(**self.params.to_dict())
        self.model = OneVsRestClassifier(self.inner_model)
        self.classes = None

    def fit(self, train_data: TensorData):
        self.classes = torch.unique(train_data.target)
        return super().fit(train_data)

    def predict_proba(self, input_data: TensorData) -> torch.Tensor:
        prediction = super().predict_proba(input_data)
        if self.model.multilabel_:
            # OneVsRest exposes only the positive probability for each binary target.
            prediction = torch.stack((1 - prediction, prediction), dim=-1)
        return prediction

    @property
    def classes_(self):
        return self.classes


class DiscriminantAnalysisImplementation(SklearnModelAdapter):
    def predict(self, input_data: TensorData) -> torch.Tensor:
        return torch.nan_to_num(super().predict(input_data))

    def predict_proba(self, input_data: TensorData) -> torch.Tensor:
        return torch.nan_to_num(super().predict_proba(input_data))

    @property
    def classes_(self):
        return self.model.classes_


class LDAImplementation(DiscriminantAnalysisImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LinearDiscriminantAnalysis(**self.params.to_dict())

    def fit(self, train_data: TensorData):
        self.check_and_correct_params()
        features, target = as_numpy(train_data.features), as_numpy(train_data.target)
        try:
            self.model.fit(features, target)
        except (ValueError, IndexError):
            # Preserve the existing fallback for singular / perfectly mapped features.
            new_solver = 'lsqr'
            self.log.debug(f'Change invalid parameter solver ({self.model.solver}) to {new_solver}')
            self.model.solver = new_solver
            self.params.update(solver=new_solver)
            self.model.fit(features, target)
        return self.model

    def check_and_correct_params(self):
        current_solver = self.params.get('solver')
        current_shrinkage = self.params.get('shrinkage')
        if current_solver == 'svd' and current_shrinkage is not None:
            self.params.update(shrinkage=None)
            self.model.shrinkage = None


class QDAImplementation(DiscriminantAnalysisImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = QuadraticDiscriminantAnalysis(**self.params.to_dict())


class ExplainableBoostingClassifierAdapter:
    """Keep numeric class labels intact at the InterpretML boundary."""

    def __init__(self, **params):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.implementation = ExplainableBoostingClassifier(**params)

    @property
    def random_state(self):
        return self.implementation.random_state

    @random_state.setter
    def random_state(self, value):
        self.implementation.random_state = value

    def fit(self, features, target):
        # InterpretML converts floating labels to strings; use integer codes
        # internally and expose the original numeric classes to the strategy.
        self.classes_, encoded = np.unique(target, return_inverse=True)
        self.implementation.fit(features, encoded)
        return self

    def predict(self, features):
        return self.classes_[self.implementation.predict(features)]

    def predict_proba(self, features):
        return self.implementation.predict_proba(features)


class ExplainableBoostingRegressorAdapter:
    def __new__(cls, **params):
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(**params)
