from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, KernelPCA, PCA
import torch

from fedot.core.constants import PCA_MIN_THRESHOLD_TS
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import flatten_if_needed
from fedot.core.operations.implementations.base import DataOperationImplementation, TensorDataOperationImplementation
from fedot.core.operations.implementations.data_operations.rules import is_spectrum_n_components_method
from fedot.core.operations.implementations.data_operations.schema import (
    validate_pca_params,
    validate_truncated_svd_params,
)
from fedot.core.operations.implementations.data_operations.tools import (
    prepare_finite_features,
    project_with_components,
    replace_projected_features,
    require_fitted_feature_width,
    resolve_pca_n_components,
    resolve_truncated_svd_n_components,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID


class PCAImplementation(TensorDataOperationImplementation):
    """PCA for TensorData via centered torch SVD.

    Args:
        params: Operation parameters. ``n_components`` may be:

            * ``int`` — fixed number of components (``1`` = one component)
            * ``float`` in ``(0, 1]`` — explained variance ratio.
              ``1.0`` is 100% variance (full rank), not sklearn's one component.
              Tuner samples ``auto`` / ``elbow`` / ``broken_stick`` and the
              same variance-ratio floats as ``truncated_svd`` (there a float
              is a fraction of ``n_features``).
            * ``'auto'`` — repository default:
              ``max(1, min(rank, n_features // 2))``. After OHE a variance
              target often keeps nearly full width; ``auto`` caps rank without
              a variance quota.
            * ``'mle'`` — Minka MLE (sklearn-compatible; needs
              ``n_samples >= n_features``)
            * ``'elbow'`` / ``'broken_stick'`` — spectrum rank selection

    Note:
        Fit drops rows with NaN (warning). Transform keeps all rows; NaN
        inputs stay NaN after projection. Resolved rank is ``n_components_``;
        the ``n_components`` hyperparameter is not overwritten.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        validated = validate_pca_params(self.params.to_dict())
        # Not params.update: that marks keys changed and the node copies them after fit.
        self.params._parameters.update(validated)

        self.mean_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.explained_variance_ratio_: Optional[torch.Tensor] = None
        self.n_components_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, data: TensorData):
        """Fit PCA on finite samples of ``data.features``.

        Args:
            data: Training TensorData.

        Returns:
            Self.
        """
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

        # Full thin SVD: needed for variance ratio / mle / elbow / broken_stick.
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
        return self

    def transform(self, data: TensorData) -> TensorData:
        """Project ``data.features`` onto fitted components.

        Args:
            data: TensorData to transform.

        Returns:
            TensorData with projected features.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError('PCAImplementation is not fitted yet.')

        features = flatten_if_needed(data.features)
        require_fitted_feature_width(features, self.n_features_, 'PCA')
        if self.n_features_ is not None and self.n_features_ <= 1:
            projected = features
        else:
            projected = project_with_components(features, self.components_, mean=self.mean_)
        return replace_projected_features(data, projected)


class TruncatedSVDImplementation(TensorDataOperationImplementation):
    """Truncated SVD for TensorData (no centering).

    Args:
        params: Operation parameters. ``n_components`` may be:

            * ``int`` — fixed number of components (``1`` = one component)
            * ``float`` in ``(0, 1]`` — fraction of ``n_features``
              (``round(fraction * n_features)``), not explained variance.
              ``1.0`` keeps all features; ``0.5`` is half the columns,
              unlike PCA where ``0.5`` is 50% variance.
              Tuner samples ``auto`` / ``elbow`` / ``broken_stick`` and the
              same fraction floats as PCA (there a float is variance share).
            * ``'auto'`` — repository default, same width cap as PCA
              (``n_features // 2``, not a variance target).
            * ``'elbow'`` / ``'broken_stick'`` — spectrum rank selection

            ``int`` / float / ``auto`` use ``torch.svd_lowrank`` when
            ``q < max_rank``; otherwise the same thin ``torch.linalg.svd``
            as spectrum modes (``elbow`` / ``broken_stick``).

            ``random_state`` is not a searchable hyperparameter; it is
            injected by ``ImplementationRandomStateHandler`` during strategy
            ``fit`` and used to seed randomized SVD.

    Note:
        Fit drops rows with NaN (warning). Transform keeps all rows; NaN
        inputs stay NaN after projection. Resolved rank is ``n_components_``;
        the ``n_components`` hyperparameter is not overwritten.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        validated = validate_truncated_svd_params(self.params.to_dict())
        # Not params.update: that marks keys changed and the node copies them after fit.
        self.params._parameters.update(validated)

        self.components_: Optional[torch.Tensor] = None
        self.n_components_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.random_state: Optional[int] = None

    def _svd_lowrank(self, matrix: torch.Tensor, q: int, niter: int):
        """Randomized SVD, seeded when ``random_state`` is set.

        ``torch.svd_lowrank`` samples from the global RNG and has no
        ``generator`` argument. Fork the RNG so repeated fits are stable
        without leaking the seed into later operations.
        """
        if self.random_state is None:
            return torch.svd_lowrank(matrix, q=q, niter=niter)

        devices = [matrix.device] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.random_state))
            if matrix.is_cuda:
                torch.cuda.manual_seed_all(int(self.random_state))
            return torch.svd_lowrank(matrix, q=q, niter=niter)

    def fit(self, data: TensorData):
        """Fit TruncatedSVD on finite samples of ``data.features``.

        Args:
            data: Training TensorData.

        Returns:
            Self.
        """
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
            # Spectrum methods need the full thin singular spectrum.
            _, singular_values, vh = torch.linalg.svd(clean, full_matrices=False)
            k = resolve_truncated_svd_n_components(
                mode,
                n_samples=self.n_samples_,
                n_features=self.n_features_,
                singular_values=singular_values,
            )
            self.n_components_ = k
            self.components_ = vh[:k].contiguous()
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

        # Full-rank request: exact SVD, same subspace as spectrum modes for this k.
        if q >= max_rank:
            _, _, vh = torch.linalg.svd(clean, full_matrices=False)
            self.components_ = vh[:k].contiguous()
        else:
            _, _, V = self._svd_lowrank(clean, q=q, niter=n_iter)
            self.components_ = V[:, :k].T.contiguous()
        self.n_components_ = k
        return self

    def transform(self, data: TensorData) -> TensorData:
        """Project ``data.features`` onto fitted components.

        Args:
            data: TensorData to transform.

        Returns:
            TensorData with projected features.
        """
        if self.components_ is None:
            raise RuntimeError('TruncatedSVDImplementation is not fitted yet.')

        features = flatten_if_needed(data.features)
        require_fitted_feature_width(features, self.n_features_, 'TruncatedSVD')
        if self.n_features_ is not None and self.n_features_ <= 1:
            projected = features
        else:
            projected = project_with_components(features, self.components_)
        return replace_projected_features(data, projected)


class ComponentAnalysisImplementation(DataOperationImplementation):
    """
    Class for applying PCA and kernel PCA models from sklearn

    Args:
        params: OperationParameters with the arguments
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = None
        self.number_of_features = None
        self.number_of_samples = None

    def fit(self, input_data: InputData) -> PCA:
        """
        The method trains the PCA model

        Args:
            input_data: data with features, target and ids for PCA training

        Returns:
            trained PCA model (optional output)
        """

        self.number_of_samples, self.number_of_features = np.array(
            input_data.features).shape

        if self.number_of_features > 1:
            self.check_and_correct_params(
                is_ts_data=input_data.data_type is DataTypesEnum.ts)
            # TODO: remove a workaround by refactoring other operations in troubled pipelines (e.g. topo)
            # workaround for NaN-containing arrays during pca fitting, especially for fast_ica
            # fast_ica cannot fit with features represented by a rather sparse matrix
            features = self.preprocess_input(input_data)
            try:
                self.pca.fit(features)
            except Exception as e:
                self.log.info(
                    f'Switched from {type(self.pca).__name__} to default PCA on fit stage due to {e}')
                self.pca = PCA()
                self.pca.fit(features)

        return self.pca

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for transformation tabular data using PCA

        Args:
            input_data: data with features, target and ids for PCA applying

        Returns:
            data with transformed features attribute
        """
        features = self.preprocess_input(input_data)

        if self.number_of_features > 1:
            transformed_features = self.pca.transform(features)
        else:
            transformed_features = features

        # Update features
        transformed_features = self.postprocess_input(transformed_features)
        output_data = self._convert_to_output(input_data, transformed_features)
        self.update_column_types(output_data)
        return output_data

    @staticmethod
    def preprocess_input(input_data: InputData) -> Union[np.ndarray, pd.DataFrame]:
        return input_data.features

    @staticmethod
    def postprocess_input(transformed_features: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return transformed_features

    def check_and_correct_params(self, is_ts_data: bool = False):
        """
        Method check if number of features in data enough for ``n_components``
        parameter in PCA or not. And if not enough - fixes it
        """
        n_components = self.params.get('n_components')
        if isinstance(n_components, int):
            if n_components > self.number_of_features:
                self.params.update(n_components=self.number_of_features)
        elif n_components == 'mle':
            from dask_ml.decomposition import PCA as DaskPCA

            # Check that n_samples correctly map with n_features
            if self.number_of_samples < self.number_of_features:
                if not isinstance(self.pca, DaskPCA):
                    self.params.update(n_components=0.5)
                else:
                    self.params.update(n_components=None)
        if is_ts_data and (n_components * self.number_of_features) < PCA_MIN_THRESHOLD_TS:
            self.params.update(
                n_components=PCA_MIN_THRESHOLD_TS / self.number_of_features)

        self.pca.set_params(**self.params.to_dict())

    @staticmethod
    def update_column_types(output_data: OutputData) -> OutputData:
        """
        Update column types after applying PCA operations
        """

        predict = np.asarray(output_data.predict)
        n_cols = 1 if predict.ndim <= 1 else predict.shape[1]

        if output_data.supplementary_data.col_type_ids is None:
            output_data.supplementary_data.col_type_ids = {}
        output_data.supplementary_data.col_type_ids['features'] = np.array(
            [TYPE_TO_ID[float]] * n_cols)
        return output_data


class DaskPCAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying PCA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        from dask_ml.decomposition import PCA as DaskPCA

        super().__init__(params)
        if not self.params:
            # Default parameters
            default_params = {'svd_solver': 'full', 'n_components': 'mle'}
            self.params.update(**default_params)
        self.pca = DaskPCA(**self.params.to_dict())
        self.number_of_features = None

    @staticmethod
    def preprocess_input(input_data: InputData) -> Union[np.ndarray, pd.DataFrame]:
        import dask.array as da
        from dask_ml.utils import check_array

        features = input_data.features
        if isinstance(input_data.features, pd.DataFrame):
            features = features.values
        features = check_array(features, accept_sparse=True, dtype=None)
        if not isinstance(features, da.Array):
            features = da.array(features)
        return features

    @staticmethod
    def postprocess_input(transformed_features):
        return transformed_features.compute()


class KernelPCAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying kernel PCA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = KernelPCA(**self.params.to_dict())


class FastICAImplementation(ComponentAnalysisImplementation):
    """
    Class for applying FastICA from sklearn

    Args:
        params: OperationParameters with the hyperparameters
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.pca = FastICA(**self.params.to_dict())
