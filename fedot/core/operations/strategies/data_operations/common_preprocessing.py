import warnings
from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.operations.implementations.data_operations.tabular import (
    NormalizationImplementation, PolyFeaturesImplementation, ScalingImplementation,
)
from fedot.core.operations.implementations.data_operations.features_reducing import (
    PCAImplementation,
    TruncatedSVDImplementation,
    KernelPCAImplementation,
    FastICAImplementation,
    DaskPCAImplementation,
)
from fedot.core.operations.implementations.data_operations.sklearn_selectors import (
    LinearClassFSImplementation, LinearRegFSImplementation,
    NonLinearClassFSImplementation, NonLinearRegFSImplementation,
)
from fedot.core.operations.implementations.data_operations.sklearn_filters import (
    IsolationForestClassImplementation, IsolationForestRegImplementation,
    LinearRegRANSACImplementation, NonLinearRegRANSACImplementation,
)
from fedot.core.operations.implementations.data_operations.resampling import ResampleImplementation
from fedot.core.operations.implementations.data_operations.decompose import (
    DecomposerClassImplementation, DecomposerRegImplementation,
)
from fedot.core.operations.implementations.data_operations.topological.fast_topological_extractor import (
    TopologicalFeaturesImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class PreprocessingDataOperationStrategy(EvaluationStrategy):
    """TensorData dispatcher for general fit/transform feature operations.

    Implementations own algorithms, fitted state and feature metadata. Training
    calls transform_for_fit; inference calls transform. Optional multi-step
    preprocessing keeps its separate service strategy.
    """

    _operations_by_types = {
        'pca': PCAImplementation,
        'truncated_svd': TruncatedSVDImplementation,
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'dask_pca': DaskPCAImplementation,
        'kernel_pca': KernelPCAImplementation,
        'poly_features': PolyFeaturesImplementation,
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
        'rfe_lin_reg': LinearRegFSImplementation,
        'rfe_non_lin_reg': NonLinearRegFSImplementation,
        'ransac_lin_reg': LinearRegRANSACImplementation,
        'ransac_non_lin_reg': NonLinearRegRANSACImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation,
        'isolation_forest_class': IsolationForestClassImplementation,
        'resample': ResampleImplementation,
        'decompose': DecomposerRegImplementation,
        'class_decompose': DecomposerClassImplementation,
        'fast_ica': FastICAImplementation,
        'topological_features': TopologicalFeaturesImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        implementation = self.operation_impl(self.params_for_fit)
        with warnings.catch_warnings(), ImplementationRandomStateHandler(implementation=implementation):
            warnings.simplefilter('ignore', RuntimeWarning)
            implementation.fit(train_data)
        return implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        return trained_operation.transform(predict_data)

    def predict_for_fit(self, trained_operation, predict_data: TensorData) -> TensorData:
        return trained_operation.transform_for_fit(predict_data)
