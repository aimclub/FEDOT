from dataclasses import replace
from typing import Optional

import numpy as np
import sklearn
import torch
from pkg_resources import parse_version
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FilterImplementation(TensorDataOperationImplementation):
    """CPU sklearn filter with device-local, training-only row selection."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.inner_model = None
        self.operation = None

    def transform(self, data: TensorData) -> TensorData:
        return replace(data, predict=None)

    def transform_for_fit(self, data: TensorData) -> TensorData:
        mask = self.operation.inlier_mask_
        if mask is None:
            self.log.info("Filtering Algorithm: didn't fit correctly. Return all objects")
            return self.transform(data)
        mask = as_torch(mask, device=data.device)
        return self.transform(data.subset_rows(torch.nonzero(mask, as_tuple=True)[0]))


class RegRANSACImplementation(FilterImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.max_iter = 10
        self.min_inliers_ratio = 0.4

    def fit(self, data: TensorData):
        target = data.target if data.target.is_floating_point() else data.target.to(torch.float64)
        threshold_step = (target - target.mean()).abs().mean().item() / self.max_iter
        residual_threshold = self.params.get('residual_threshold')
        if residual_threshold is None:
            # sklearn's default is the median absolute deviation. Make it explicit
            # so the existing adaptive threshold can also be increased after a failed fit.
            median = target.flatten().quantile(0.5)
            residual_threshold = (target - median).abs().flatten().quantile(0.5).item()
        features = as_numpy(data.features)
        target_array = as_numpy(data.target.squeeze())
        for _ in range(self.max_iter):
            try:
                self.operation.inlier_mask_ = None
                self.params.update(residual_threshold=residual_threshold)
                self.operation.residual_threshold = residual_threshold
                self.operation.fit(features, target_array)
                mask = as_torch(self.operation.inlier_mask_)
                if mask.to(torch.float64).mean().item() >= self.min_inliers_ratio:
                    return self
            # Consensus trials can fail for degenerate subsets; preserve the
            # existing retry policy and the final unfiltered fallback.
            except (ValueError, np.linalg.LinAlgError):
                pass
            self.log.info(f'RANSAC: increased residual_threshold by {threshold_step}')
            residual_threshold += threshold_step
        self.operation.inlier_mask_ = None
        return self


class LinearRegRANSACImplementation(RegRANSACImplementation):
    """RANSAC with the historical sklearn scaling + linear regression estimator."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # Scaling is fitted inside each consensus trial, on that trial's subset.
        self.inner_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
        if parse_version(sklearn.__version__) < parse_version('1.1.0'):
            self.operation = RANSACRegressor(base_estimator=self.inner_model, **self.params.to_dict())
        else:
            self.operation = RANSACRegressor(estimator=self.inner_model, **self.params.to_dict())


class NonLinearRegRANSACImplementation(RegRANSACImplementation):
    """RANSAC with the historical sklearn DecisionTreeRegressor estimator."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.inner_model = DecisionTreeRegressor()
        if parse_version(sklearn.__version__) < parse_version('1.1.0'):
            self.operation = RANSACRegressor(base_estimator=self.inner_model, **self.params.to_dict())
        else:
            self.operation = RANSACRegressor(estimator=self.inner_model, **self.params.to_dict())


class IsolationForestRegImplementation(FilterImplementation):
    """Isolation Forest filtering for regression; inference keeps every row."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.operation = IsolationForest(**self.params.to_dict())

    def fit(self, data: TensorData):
        # sklearn ignores y for this unsupervised estimator; keep target on its device.
        self.operation.fit(as_numpy(data.features))
        return self

    def _get_inlier_mask(self, data: TensorData) -> torch.Tensor:
        prediction = self.operation.predict(as_numpy(data.features))
        return as_torch(prediction, device=data.device) == 1

    def transform_for_fit(self, data: TensorData) -> TensorData:
        mask = self._get_inlier_mask(data)
        return self.transform(data.subset_rows(torch.nonzero(mask, as_tuple=True)[0]))


class IsolationForestClassImplementation(IsolationForestRegImplementation):
    """Isolation Forest with the existing per-class retention safeguard."""

    @staticmethod
    def _is_inlier_mask_correct(input_targets: torch.Tensor, modified_targets: torch.Tensor) -> bool:
        classes, counts = torch.unique(input_targets, return_counts=True)
        remaining, remaining_counts = torch.unique(modified_targets, return_counts=True)
        return torch.equal(classes, remaining) and bool((counts <= 5 * remaining_counts).all())

    def transform_for_fit(self, data: TensorData) -> TensorData:
        mask = self._get_inlier_mask(data)
        if self._is_inlier_mask_correct(data.target, data.target[mask]):
            return self.transform(data.subset_rows(torch.nonzero(mask, as_tuple=True)[0]))
        return self.transform(data)
