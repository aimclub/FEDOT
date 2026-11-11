from dataclasses import replace
from typing import Optional

import torch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FeatureSelectionImplementation(TensorDataOperationImplementation):
    """Fit sklearn RFE on CPU; apply its selected columns in Torch on any device."""

    estimator_type = None

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.inner_model = self.estimator_type()
        rfe_params = {key: value for key, value in self.params.to_dict().items()
                      if key in ('n_features_to_select', 'step')}
        self.operation = RFE(estimator=self.inner_model, **rfe_params)
        self.remain_features_mask = None
        self.selected_columns = None

    def fit(self, data: TensorData):
        width = data.features.shape[1] if data.features.ndim > 1 else 1
        if width == 1:
            # RFE requires at least two columns; the historical operation is an identity here.
            self.remain_features_mask = torch.ones(1, dtype=torch.bool)
        else:
            target = data.target
            # LogisticRegression cannot fit multiple targets. Preserve the old
            # first-target fallback without catching unrelated sklearn errors.
            # Trees and linear regression retain all targets, as before.
            if isinstance(self.inner_model, LogisticRegression) and target.ndim == 2:
                target = target[:, 0]
            self.operation.fit(as_numpy(data.features), as_numpy(target))
            self.remain_features_mask = as_torch(self.operation.support_, dtype=torch.bool, copy=True)
        self.selected_columns = torch.nonzero(self.remain_features_mask, as_tuple=True)[0]
        return self

    def transform(self, data: TensorData) -> TensorData:
        if data.features.ndim == 1:
            return replace(data, predict=None, fingerprint=None)
        return data.subset_features(self.selected_columns)


class LinearRegFSImplementation(FeatureSelectionImplementation):
    """Recursive feature elimination using sklearn LinearRegression."""

    estimator_type = LinearRegression


class NonLinearRegFSImplementation(FeatureSelectionImplementation):
    """Recursive feature elimination using sklearn DecisionTreeRegressor."""

    estimator_type = DecisionTreeRegressor


class LinearClassFSImplementation(FeatureSelectionImplementation):
    """Recursive feature elimination using sklearn LogisticRegression."""

    estimator_type = LogisticRegression


class NonLinearClassFSImplementation(FeatureSelectionImplementation):
    """Recursive feature elimination using sklearn DecisionTreeClassifier."""

    estimator_type = DecisionTreeClassifier
