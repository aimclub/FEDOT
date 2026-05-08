import numpy as np
import pytest
import torch

from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OPTIONAL_MAPPING
from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStepEnum,
                                                          ImputationMethodEnum,
                                                          ScalingMethodEnum,
                                                          FilteringMethodEnum,
                                                          EncodingMethodEnum)
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.optional_planner import build_optional_plan


@pytest.mark.unit
def test_build_optional_plan():
    """Test optional plan construction from explicit imputation params.

    Checks that `build_optional_plan` returns a PreprocessingPlan containing exactly
    one mean-imputation step for the configured feature index."""
    tensor = torch.Tensor([[1, float('nan'), 3], [4, 5, 6]])
    data = TensorDataCreator.create(tensor, "cpu")
    pipeline = None

    optional_steps = {
        PreprocessingStepEnum.imputation: [
            {
                'method': ImputationMethodEnum.mean,
                'features_idx': [0],
                'step_args': None
            }
        ]
    }

    optional_plan = build_optional_plan(data, optional_steps)

    assert isinstance(optional_plan, PreprocessingPlan)
    assert len(optional_plan.steps) == 1


@pytest.mark.unit
def test_mean_imputation():
    """Test direct mean imputation handler behavior.

    Checks that NaN in column 1 is replaced with the mean of valid values `2` and
    `8`, producing value `5`."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    preprocessor = PREPROCESSING_OPTIONAL_MAPPING[PreprocessingStepEnum.imputation][ImputationMethodEnum.mean]()
    preprocessed_data = preprocessor.fit_transform(td, [1])
    assert preprocessed_data.features[1, 1] == 5


@pytest.mark.unit
def test_preprocessing_plan_imputation():
    """Test default optional imputation through `OptionalTabularService`.

    Checks that automatic imputation returns PreparedData and fills the missing value
    in column 1 with the inferred mean/median value `5`."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, {PreprocessingStepEnum.imputation: None})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


@pytest.mark.unit
def test_preprocessing_plan_mode_imputation():
    """Test mode imputation through optional preprocessing plan.

    Checks that NaN in column 1 is replaced by the most frequent value `2` and the
    service returns PreparedData."""
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mode,
                                           "features_idx": [1],
                                            "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 2


@pytest.mark.unit
def test_preprocessing_plan_mean_imputation():
    """Test configured mean imputation through optional service.

    Checks that explicit mean-imputation strategy fills column 1 NaN with `5` and
    keeps the result wrapped as PreparedData."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mean,
                                           "features_idx": [1],
                                            "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


@pytest.mark.unit
def test_preprocessing_plan_constant_imputation():
    """Test constant imputation through optional service.

    Checks that NaN in column 1 is replaced by configured constant `3` and the output
    is PreparedData."""
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.constant,
                                           "features_idx": [1],
                                            "step_args": {"constant": 3}}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 3


@pytest.mark.unit
def test_preprocessing_plan_delete_raw_imputation():
    """Test row deletion imputation strategy.

    Checks that rows containing NaN in selected column are removed from both features
    and target, leaving two aligned samples."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorDataCreator.create(X, backend_name="cpu")
    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.delete_raw,
                                           "features_idx": [1],
                                            "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features.shape[0] == 2
    assert preprocessed_data.target.shape[0] == 2


@pytest.mark.unit
def test_preprocessing_minmax_scaling():
    """Test min-max scaling for a selected tabular column.

    Compares non-NaN values in column 1 with manual `(x - min) / (max - min)` result,
    checks NaN preservation, and verifies unselected column 0 is unchanged."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorDataCreator.create(X, backend_name="cpu")

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.min_max,
                "features_idx": [1],
                "step_args": None
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    # col 1: [2, nan, 8]
    # min = 2, max = 8
    # (x - 2) / (8 - 2)

    expected_col = np.array([
        (2 - 2) / 6,
        np.nan,
        (8 - 2) / 6
    ], dtype=np.float32)

    assert np.allclose(result[[0, 2], 1], expected_col[[0, 2]], atol=1e-6)
    assert np.isnan(result[1, 1])
    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


@pytest.mark.unit
def test_preprocessing_standard_scaling():
    """Test standard scaling for a selected tabular column.

    Compares column 1 with manual z-score using valid values `[2, 8]`, checks NaN
    preservation, and verifies unselected column 0 is unchanged."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorDataCreator.create(X, backend_name="cpu")

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.standard,
                "features_idx": [1],
                "step_args": None
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    # col 1: [2, nan, 8]
    valid = np.array([2, 8], dtype=np.float32)

    mean = valid.mean()
    std = valid.std()  # ddof=0 как в sklearn

    expected_col = np.array([
        (2 - mean) / std,
        np.nan,
        (8 - mean) / std
    ], dtype=np.float32)

    assert np.allclose(result[[0, 2], 1], expected_col[[0, 2]], atol=1e-6)
    assert np.isnan(result[1, 1])
    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


@pytest.mark.unit
def test_preprocessing_robust_scaling():
    """Test robust scaling by median and interquartile range.

    Compares selected column values with manual `(x - median) / IQR` calculation,
    keeps NaN in place, and checks unselected feature values are unchanged."""
    X = np.array([
        [1, 10, 3],
        [2, np.nan, 6],
        [3, 20, 9],
        [4, 30, 12],
        [5, 40, 15],
    ], dtype=np.float32)

    td = TensorDataCreator.create(X, backend_name="cpu")

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.robust,
                "features_idx": [1],
                "step_args": None,
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    valid = np.array([10, 20, 30, 40], dtype=np.float32)
    median = np.median(valid)
    q25 = np.quantile(valid, 0.25)
    q75 = np.quantile(valid, 0.75)
    iqr = q75 - q25

    expected_col = np.array([
        (10 - median) / iqr,
        np.nan,
        (20 - median) / iqr,
        (30 - median) / iqr,
        (40 - median) / iqr,
    ], dtype=np.float32)

    assert np.allclose(result[[0, 2, 3, 4], 1], expected_col[[0, 2, 3, 4]], atol=1e-6)
    assert np.isnan(result[1, 1])

    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


@pytest.mark.unit
def test_imputation_scaling():
    """Test sequential optional imputation followed by scaling.

    Checks that constant imputation first converts NaN to `3`, then min-max scaling
    produces the expected `[0, 1/6, 1]` column with no NaNs."""
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorDataCreator.create(X, backend_name="cpu")

    service = OptionalTabularService()
    strategy = {
        PreprocessingStepEnum.imputation: [{
            "method": ImputationMethodEnum.constant,
            "features_idx": [1],
            "step_args": {"constant": 3}
        }],
        PreprocessingStepEnum.scaling: [{
            "method": ScalingMethodEnum.min_max,
            "features_idx": [1],
            "step_args": {}
        }]
    }

    preprocessed_data = service.fit_transform(td, strategy)

    assert isinstance(preprocessed_data, PreparedData)

    result = preprocessed_data.features.numpy()

    # col 1:
    # [2, 3, 8]
    # min = 2, max = 8
    # scaling → (x - 2) / 6

    expected_col = np.array([
        (2 - 2) / 6,   # 0
        (3 - 2) / 6,   # 1/6
        (8 - 2) / 6    # 1
    ], dtype=np.float32)

    assert np.allclose(result[:, 1], expected_col, atol=1e-6)
    assert not np.isnan(result[:, 1]).any()
    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


@pytest.mark.unit
def test_encoding_autoscaling_imputation():
    """Test optional preprocessing after obligatory categorical encoding.

    Checks that categorical column is label-encoded, numeric column is auto-scaled,
    configured imputation fills NaN with `3`, and another numeric feature changes due
    to scaling."""
    X = np.array([
        [1, 2, "A", 3],
        [4, np.nan, "B", 6],
        [7, 8, "C", 9]
    ], dtype=object)

    td = TensorDataCreator.create(X, backend_name="cpu")

    strategy = {
        PreprocessingStepEnum.scaling: None,
        PreprocessingStepEnum.imputation: [{
            "method": ImputationMethodEnum.constant,
            "features_idx": [1],
            "step_args": {"constant": 3}
        }],
    }

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, strategy)

    assert isinstance(preprocessed_data, PreparedData)

    result = preprocessed_data.features.numpy()

    # col 1: [2, nan, 8]
    # min = 2, max = 8
    # (x - 2) / (8 - 2)

    expected_col = np.array([
        (2 - 2) / 6,
        np.nan,
        (8 - 2) / 6
    ], dtype=np.float32)

    assert not np.allclose(result[:, 0].astype(np.float32), X[:, 0].astype(np.float32))
    assert np.allclose(result[[0, 2], 1], expected_col[[0, 2]], atol=1e-6)
    assert result[1, 1] == 3
    assert np.allclose(result[:, 2], np.array([0, 1, 2], dtype=np.float32), atol=1e-6)


@pytest.mark.unit
def test_ohe_encoding_imputation_uses_original_indices():
    """Test optional preprocessing index mapping after one-hot encoding.

    Checks that optional preprocessing still targets the original numeric column
    after OHE moves the encoded categorical column to the end of the feature matrix.
    """
    X = np.array([
        [1, "A", 2, 3],
        [4, "B", np.nan, 6],
        [7, "C", 8, 9]
    ], dtype=object)

    encoding_strategy = [{
        "method": EncodingMethodEnum.ohe,
        "features_idx": [1]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        encoding_strategy=encoding_strategy
    )

    strategy = {
        PreprocessingStepEnum.imputation: [{
            "method": ImputationMethodEnum.constant,
            "features_idx": [2],
            "step_args": {"constant": 3}
        }],
    }

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, strategy)

    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features.shape[1] == 5

    result = preprocessed_data.features.numpy()

    reference_result = np.array([
        [1, 2, 1, 0, 0],
        [4, 3, 0, 1, 0],
        [7, 8, 0, 0, 1],
    ], dtype=np.float32)

    assert np.allclose(result, reference_result, atol=1e-6)


@pytest.mark.unit
def test_ohe_encoding_imputation_uses_original_feature_names():
    """Test optional preprocessing name mapping after one-hot encoding."""
    X = np.array([
        [1, "A", 2, 3],
        [4, "B", np.nan, 6],
        [7, "C", 8, 9]
    ], dtype=object)
    columns = ["id", "category", "value", "target"]

    encoding_strategy = [{
        "method": EncodingMethodEnum.ohe,
        "features_idx": ["category"]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        features_names=columns,
        encoding_strategy=encoding_strategy
    )

    strategy = {
        PreprocessingStepEnum.imputation: [{
            "method": ImputationMethodEnum.constant,
            "features_idx": ["value"],
            "step_args": {"constant": 3}
        }],
    }

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(td, strategy)

    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features.shape[1] == 5

    result = preprocessed_data.features.numpy()

    reference_result = np.array([
        [1, 2, 1, 0, 0],
        [4, 3, 0, 1, 0],
        [7, 8, 0, 0, 1],
    ], dtype=np.float32)

    assert np.allclose(result, reference_result, atol=1e-6)


@pytest.mark.unit
def test_preprocessing_clipping():
    """Test quantile clipping filter for outlier handling.

    Compares selected column with manual clipping between 10th and 90th percentiles,
    verifies NaN preservation, and checks unselected column remains unchanged."""
    X = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, np.nan, 300],
        [4, 40, 400],
        [5, 50, 500],
        [6, 1000, 600],
    ], dtype=np.float32)

    td = TensorDataCreator.create(X, backend_name="cpu")

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.filtering: [{
                "method": FilteringMethodEnum.quantile,
                "features_idx": [1],
                "step_args": {"quantile_range": (10.0, 90.0)},
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    x = X[:, 1]
    valid = x[~np.isnan(x)]

    lower = np.quantile(valid, 0.10)
    upper = np.quantile(valid, 0.90)

    expected_col = x.copy()
    valid_mask = ~np.isnan(expected_col)
    expected_col[valid_mask] = np.clip(expected_col[valid_mask], lower, upper)

    compare_mask = ~np.isnan(expected_col)
    assert np.allclose(result[compare_mask, 1], expected_col[compare_mask], atol=1e-6)
    assert np.isnan(result[2, 1])

    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


@pytest.mark.unit
def test_custom_preprocessing():
    """Test custom optional preprocessing handlers.

    Checks that a custom zero imputer fills NaNs in column 1 with `0`, a custom root
    constant imputer fills column 2 with `sqrt(100)=10`, and both columns match the
    reference arrays."""
    X = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, np.nan, 300],
        [4, 40, np.nan],
        [5, 50, 500],
        [6, 1000, 600],
    ], dtype=np.float32)
    y = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)

    td = TensorDataCreator.create(X, target=y, backend_name="cpu")

    from typing import Optional, Sequence

    class CustomZeroImputer(AbstractPreprocessingHandler):
        def __init__(self):
            self.features_idx: Optional[Sequence[int]] = None

        def fit(self, data: PreparedData, features_idx: Sequence[int]):
            self.features_idx = features_idx
            return self

        def transform(self, data: PreparedData) -> PreparedData:
            if self.features_idx is None:
                raise RuntimeError("ConstantImputation is not fitted yet.")

            for col_idx in self.features_idx:
                column = data.features[:, col_idx]
                data.features[:, col_idx] = torch.where(
                    torch.isnan(column),
                    torch.tensor(0, device=data.features.device, dtype=data.features.dtype),
                    column
                )

            return data

    class CustomRootConstantImputer(AbstractPreprocessingHandler):
        def __init__(self, constant: float = 0.0):
            self.constant = constant

            self.features_idx: Optional[Sequence[int]] = None
            self.root_constant: Optional[float] = None

        def fit(self, data: PreparedData, features_idx: Sequence[int]):
            self.features_idx = features_idx
            self.root_constant = self.constant ** 0.5
            return self

        def transform(self, data: PreparedData) -> PreparedData:
            if self.features_idx is None:
                raise RuntimeError("ConstantImputation is not fitted yet.")

            for col_idx in self.features_idx:
                column = data.features[:, col_idx]
                data.features[:, col_idx] = torch.where(
                    torch.isnan(column),
                    torch.tensor(self.root_constant, device=data.features.device, dtype=data.features.dtype),
                    column
                )

            return data

    service = OptionalTabularService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.custom: [{
                "method": 'ZeroImputer',
                "features_idx": [1],
                "implementation": CustomZeroImputer,
                "step_args": None,
            },
                {
                "method": 'RootConstantImputer',
                "features_idx": [2],
                "implementation": CustomRootConstantImputer,
                "step_args": {"constant": 100},
            },
            ]
        }
    )

    result = preprocessed_data.features.numpy()
    assert np.allclose(result[:, 1], np.array([10, 20, 0, 40, 50, 1000], dtype=np.float32), atol=1e-6)
    assert np.allclose(result[:, 2], np.array([100, 200, 300, 10, 500, 600], dtype=np.float32), atol=1e-6)
