from typing import Optional

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import flatten_if_needed
from fedot.core.operations.evaluation.abstract_node import TensorDataOperationImplementation
from fedot.core.operations.evaluation.operation_implementations.rules import (
    is_spectrum_n_components_method,
)
from fedot.core.operations.evaluation.operation_implementations.schema import (
    validate_pca_params,
    validate_truncated_svd_params,
)
from fedot.core.operations.evaluation.operation_implementations.tools import (
    prepare_finite_features,
    project_with_components,
    replace_projected_features,
    resolve_pca_n_components,
    resolve_truncated_svd_n_components,
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
            clean = prepare_finite_features(
                features, self.log, 'PCA', require_min_samples=False,
            )
            self.n_samples_ = clean.shape[0]
            self.mean_ = (
                clean.mean(dim=0) if self.n_samples_ > 0
                else features.new_zeros(self.n_features_)
            )
            self.components_ = torch.eye(
                self.n_features_, device=features.device, dtype=features.dtype,
            )
            self.explained_variance_ratio_ = torch.ones(
                self.n_features_,
                device=features.device,
                dtype=features.dtype,
            )
            self.n_components_ = self.n_features_
            return self

        clean = prepare_finite_features(features, self.log, 'PCA')
        self.n_samples_ = clean.shape[0]

        self.mean_ = clean.mean(dim=0)
        centered = clean - self.mean_

        # Full SVD on centered matrix: X = U S V^T, components = rows of V^T
        # (needed when n_components is a variance ratio / mle / spectrum method).
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        explained_variance = (singular_values ** 2) / max(self.n_samples_ - 1, 1)
        total_var = explained_variance.sum()
        if float(total_var) > 0:
            self.explained_variance_ratio_ = explained_variance / total_var
        else:
            self.explained_variance_ratio_ = torch.zeros_like(explained_variance)

        n_components = resolve_pca_n_components(
            self.params.get('n_components'),
            n_samples=self.n_samples_,
            n_features=self.n_features_,
            explained_variance_ratio=self.explained_variance_ratio_,
            singular_values=singular_values,
        )
        self.n_components_ = n_components
        self.components_ = vh[:n_components].contiguous()
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components].contiguous()
        self.params.update(n_components=n_components)
        return self

    def transform(self, data: TensorData) -> TensorData:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError('PCAImplementation is not fitted yet.')

        features = flatten_if_needed(data.features)
        if self.n_features_ is not None and self.n_features_ <= 1:
            projected = features
        else:
            projected = project_with_components(features, self.components_, mean=self.mean_)
        return replace_projected_features(data, projected)


class TruncatedSVDImplementation(TensorDataOperationImplementation):
    """Truncated SVD for TensorData via ``torch.svd_lowrank`` (no centering).

    Unlike PCA, does not center features. ``n_components`` may be a positive int,
    a float feature-fraction in ``(0, 1]``, ``auto`` (half-feature budget), or a
    spectrum method (``elbow`` / ``broken_stick``) that needs a thin SVD first.
    Useful after OHE / high-dimensional encodings.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        validated = validate_truncated_svd_params(self.params.to_dict())
        self.params.update(
            n_components=validated['n_components'],
            n_iter=validated['n_iter'],
            n_oversamples=validated['n_oversamples'],
        )

        self.components_: Optional[torch.Tensor] = None
        self.n_components_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, data: TensorData):
        features = flatten_if_needed(data.features)
        self.n_features_ = features.shape[1]

        if self.n_features_ <= 1:
            clean = prepare_finite_features(
                features, self.log, 'TruncatedSVD', require_min_samples=False,
            )
            self.n_samples_ = clean.shape[0]
            self.components_ = torch.eye(
                self.n_features_, device=features.device, dtype=features.dtype,
            )
            self.n_components_ = self.n_features_
            return self

        clean = prepare_finite_features(features, self.log, 'TruncatedSVD')
        self.n_samples_ = clean.shape[0]

        mode = self.params.get('n_components')
        if is_spectrum_n_components_method(mode):
            # Spectrum rank selection needs the full thin singular spectrum.
            _, singular_values, vh = torch.linalg.svd(clean, full_matrices=False)
            k = resolve_truncated_svd_n_components(
                mode,
                n_samples=self.n_samples_,
                n_features=self.n_features_,
                singular_values=singular_values,
            )
            self.n_components_ = k
            self.components_ = vh[:k].contiguous()
            self.params.update(n_components=k)
            return self

        k = resolve_truncated_svd_n_components(
            mode,
            n_samples=self.n_samples_,
            n_features=self.n_features_,
        )
        max_rank = min(self.n_samples_, self.n_features_)
        n_iter = int(self.params.get('n_iter', 5))
        n_oversamples = int(self.params.get('n_oversamples', 10))
        q = min(k + n_oversamples, max_rank)

        # Randomized / truncated SVD: only approximate rank-q factors, keep k.
        _, _, V = torch.svd_lowrank(clean, q=q, niter=n_iter)
        self.n_components_ = k
        self.components_ = V[:, :k].T.contiguous()
        self.params.update(n_components=k)
        return self

    def transform(self, data: TensorData) -> TensorData:
        if self.components_ is None:
            raise RuntimeError('TruncatedSVDImplementation is not fitted yet.')

        features = flatten_if_needed(data.features)
        if self.n_features_ is not None and self.n_features_ <= 1:
            projected = features
        else:
            projected = project_with_components(features, self.components_)
        return replace_projected_features(data, projected)
