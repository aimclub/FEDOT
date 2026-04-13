import numpy as np
import torch

from fedot.preprocessing.mapping import PREPROCESSING_OPTIONAL_MAPPING
from fedot.preprocessing.preprocessor_types import (PreprocessingStepEnum, 
                                                    ImputationMethodEnum, 
                                                    ScalingMethodEnum,
                                                    FilteringMethodEnum)
from fedot.preprocessing.service import OtionalPreprocessingService
from fedot.core.data.prepared_data import PreparedData
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.planner import build_optional_plan, PreprocessingPlan


def test_build_optional_plan():
    tensor = torch.Tensor([[1, float('nan'), 3], [4, 5, 6]])
    data = TensorData.create(tensor, "cpu")
    pipeline = None

    optional_steps ={
        PreprocessingStepEnum.imputation: [
            {
                'method': ImputationMethodEnum.mean, 
                'features_idx': [0],
                'step_args': None
            }
        ]
    }

    optional_plan = build_optional_plan(data, pipeline, optional_steps)
    
    assert isinstance(optional_plan, PreprocessingPlan)
    assert len(optional_plan.steps) == 1


def test_mean_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    preprocessor = PREPROCESSING_OPTIONAL_MAPPING[PreprocessingStepEnum.imputation][ImputationMethodEnum.mean]()
    preprocessed_data = preprocessor.fit_transform(td, [1])
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {PreprocessingStepEnum.imputation: None})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_mode_imputation():
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mode, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 2


def test_preprocessing_plan_mean_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mean, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_constant_imputation():
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.constant, 
                                           "features_idx": [1],
                                           "step_args": {"constant": 3}}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 3


def test_preprocessing_plan_delete_raw_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.delete_raw, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features.shape[0] == 2
    assert preprocessed_data.target.shape[0] == 2


def test_preprocessing_minmax_scaling():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
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


def test_preprocessing_standard_scaling():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
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


def test_preprocessing_robust_scaling():
    X = np.array([
        [1, 10, 3],
        [2, np.nan, 6],
        [3, 20, 9],
        [4, 30, 12],
        [5, 40, 15],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
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


def test_imputation_scaling():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
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

    preprocessed_data = service.fit_transform(td, None, strategy)

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


def test_preprocessing_seasonal_normalization():
    X = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 13, 400],
        [5, np.nan, 500],
        [6, 33, 600],
        [7, 16, 700],
        [8, 26, 800],
        [9, 36, 900],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.seasonal,
                "features_idx": [1],
                "step_args": {"period": 3},
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected_col = np.full(X.shape[0], np.nan, dtype=np.float32)

    # phase 0: idx [0, 3, 6] -> [10, 13, 16]
    phase0 = np.array([10, 13, 16], dtype=np.float32)
    mean0 = phase0.mean()
    std0 = phase0.std()
    expected_col[[0, 3, 6]] = (phase0 - mean0) / std0

    # phase 1: idx [1, 4, 7] -> [20, nan, 26]
    phase1_valid = np.array([20, 26], dtype=np.float32)
    mean1 = phase1_valid.mean()
    std1 = phase1_valid.std()
    expected_col[1] = (20 - mean1) / std1
    expected_col[4] = np.nan
    expected_col[7] = (26 - mean1) / std1

    # phase 2: idx [2, 5, 8] -> [30, 33, 36]
    phase2 = np.array([30, 33, 36], dtype=np.float32)
    mean2 = phase2.mean()
    std2 = phase2.std()
    expected_col[[2, 5, 8]] = (phase2 - mean2) / std2

    valid_mask = ~np.isnan(expected_col)
    assert np.allclose(result[valid_mask, 1], expected_col[valid_mask], atol=1e-6)
    assert np.isnan(result[4, 1])

    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


def test_preprocessing_rolling_normalization():
    X = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, np.nan, 300],
        [4, 40, 400],
        [5, 50, 500],
        [6, 60, 600],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.rolling,
                "features_idx": [1],
                "step_args": {"window_size": 3},
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    x = X[:, 1]
    expected_col = []

    for t in range(len(x)):
        start = max(0, t - 3 + 1)
        window = x[start:t + 1]
        valid = window[~np.isnan(window)]

        if len(valid) == 0:
            expected_col.append(np.nan)
            continue

        mean = valid.mean()
        std = valid.std()

        if np.isnan(x[t]):
            expected_col.append(np.nan)
        else:
            if std == 0:
                std = 1.0
            expected_col.append((x[t] - mean) / std)

    expected_col = np.array(expected_col, dtype=np.float32)

    valid_mask = ~np.isnan(expected_col)
    assert np.allclose(result[valid_mask, 1], expected_col[valid_mask], atol=1e-6)
    assert np.isnan(result[2, 1])

    assert np.allclose(result[:, 0], X[:, 0], atol=1e-6)


def test_encoding_autoscaling_imputation():
    X = np.array([
        [1, 2, "A", 3],
        [4, np.nan, "B", 6],
        [7, 8, "C", 9]
    ], dtype=object)

    td = TensorData.create(X, backend_name="cpu")

    strategy = {
        PreprocessingStepEnum.scaling: None,
        PreprocessingStepEnum.imputation: [{
            "method": ImputationMethodEnum.constant,
            "features_idx": [1],
            "step_args": {"constant": 3}
        }],
    }

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, strategy)

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


def test_preprocessing_clipping():
    X = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, np.nan, 300],
        [4, 40, 400],
        [5, 50, 500],
        [6, 1000, 600],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu")

    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        None,
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



if __name__ == "__main__":
    # test_encoding_autoscaling_imputation()
    # test_preprocessing_clipping()
