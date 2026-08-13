from dataclasses import replace
from typing import Optional, Union

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import drop_rows_with_nan, flatten_if_needed
from fedot.core.operations.evaluation.abstract_node import TensorDataOperationImplementation
from fedot.core.operations.evaluation.operation_implementations.rules import (
    PCA_MIN_THRESHOLD_TS,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_pca_fit_samples,
    validate_pca_params,
)
from fedot.core.operations.operation_parameters import OperationParameters


class PCAImplementation(TensorDataOperationImplementation):
    """PCA for TensorData via torch SVD (sklearn-``pca`` analogue).

    ``n_components`` is validated/completed via :func:`validate_pca_params`
    (marshmallow + ``default_operation_params.json``).

    Fit drops samples that contain NaN (with a warning). Transform keeps the
    original number of rows: NaN inputs stay NaN after projection.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        validated = validate_pca_params(self.params.to_dict())
        self.params.update(n_components=validated['n_components'])

        self.mean_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.explained_variance_ratio_: Optional[torch.Tensor] = None
        self.n_components_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, data: TensorData):
        features = flatten_if_needed(data.features)
        self.n_features_ = features.shape[1]

        if self.n_features_ <= 1:
            clean, _ = drop_rows_with_nan(features)
            self.n_samples_ = clean.shape[0]
            self.mean_ = clean.mean(dim=0) if self.n_samples_ > 0 else features.new_zeros(self.n_features_)
            self.components_ = torch.eye(
                self.n_features_,
                device=features.device,
                dtype=features.dtype,
            )
            self.explained_variance_ratio_ = torch.ones(
                self.n_features_,
                device=features.device,
                dtype=features.dtype,
            )
            self.n_components_ = self.n_features_
            return self

        clean, n_dropped = drop_rows_with_nan(features)
        if n_dropped:
            self.log.warning(
                f'PCA fit: dropping {n_dropped} sample(s) with NaN; '
                f'they are not used for fitting. Transform still returns all rows '
                f'(NaN inputs remain NaN after projection).'
            )
        self.n_samples_ = clean.shape[0]
        validate_pca_fit_samples(self.n_samples_)

        self.mean_ = clean.mean(dim=0)
        centered = clean - self.mean_

        # Full SVD on centered matrix: X = U S V^T, components = rows of V^T
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        explained_variance = (singular_values ** 2) / max(self.n_samples_ - 1, 1)
        total_var = explained_variance.sum()
        if float(total_var) > 0:
            self.explained_variance_ratio_ = explained_variance / total_var
        else:
            self.explained_variance_ratio_ = torch.zeros_like(explained_variance)

        n_components = self._resolve_n_components(self.params.get('n_components'))
        self.n_components_ = n_components
        self.components_ = vh[:n_components].contiguous()
        self.params.update(n_components=n_components)
        return self

    def transform(self, data: TensorData) -> TensorData:
        features = flatten_if_needed(data.features)
        if self.n_features_ is not None and self.n_features_ <= 1:
            projected = features
        else:
            # Keep all rows; NaN entries propagate through the linear map.
            centered = features - self.mean_.to(device=features.device, dtype=features.dtype)
            components = self.components_.to(device=features.device, dtype=features.dtype)
            projected = centered @ components.T

        return replace(
            data,
            features=projected,
            categorical_idx=[],
            numerical_idx=list(range(projected.shape[1])),
            features_names=None,
            fingerprint=None,
        )

    def _resolve_n_components(
        self,
        n_components: Union[int, float, str],
        is_ts_data: bool = False,
    ) -> int:
        n_components = validate_pca_params({'n_components': n_components}, None)['n_components']
        max_components = max(min(self.n_samples_, self.n_features_), 1)

        if isinstance(n_components, str):
            # Schema allows only ``mle`` among strings.
            if self.n_samples_ < self.n_features_:
                n_components = 0.5
            else:
                return max(1, max_components - 1) if max_components > 1 else 1

        if isinstance(n_components, float) and n_components < 1.0:
            if is_ts_data and self.n_features_ > 0:
                if (n_components * self.n_features_) < PCA_MIN_THRESHOLD_TS:
                    n_components = PCA_MIN_THRESHOLD_TS / self.n_features_
            cumsum = torch.cumsum(self.explained_variance_ratio_, dim=0)
            hits = (cumsum >= n_components).nonzero(as_tuple=False)
            if hits.numel() == 0:
                return max_components
            return int(hits[0].item()) + 1

        resolved = int(n_components)
        if resolved > max_components:
            resolved = max_components
        return max(1, resolved)
