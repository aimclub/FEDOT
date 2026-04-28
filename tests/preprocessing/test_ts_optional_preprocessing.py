import numpy as np

from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStepEnum,
                                                    ScalingMethodEnum,
                                                    ImagePreprocessingMethodEnum,
                                                    ImputationMethodEnum)
from fedot.industrial.core.architecture.preprocessing.ts_optional_service import TSPreprocessingService
from fedot.core.data.tensordata import TensorData


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

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
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

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
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


def test_preprocessing_per_channel_normalization():
    X = np.array([
        [
            [1, 10],
            [2, 20],
            [3, 30],
        ],
        [
            [4, 40],
            [5, np.nan],
            [6, 60],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.scaling: [{
                "method": ScalingMethodEnum.standart_per_channel,
                "features_idx": None,
                "step_args": {"channels_idx": [1]},
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    channel_values = X[:, :, 1].reshape(-1)
    valid = channel_values[~np.isnan(channel_values)]

    mean = valid.mean()
    std = valid.std()
    if std == 0:
        std = 1.0

    expected_channel = np.array([
        [
            (10 - mean) / std,
            (20 - mean) / std,
            (30 - mean) / std,
        ],
        [
            (40 - mean) / std,
            np.nan,
            (60 - mean) / std,
        ],
    ], dtype=np.float32)

    valid_mask = ~np.isnan(expected_channel)
    assert np.allclose(result[:, :, 1][valid_mask], expected_channel[valid_mask], atol=1e-6)
    assert np.isnan(result[1, 1, 1])

    assert np.allclose(result[:, :, 0], X[:, :, 0], atol=1e-6)


def test_preprocessing_gamma_correction():
    X = np.array([
        [
            [1, 0.0],
            [2, 0.25],
            [3, 0.5],
            [4, 1.0],
        ],
        [
            [5, 0.75],
            [6, np.nan],
            [7, 0.04],
            [8, 0.81],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.image_preprocessing: [{
                "method": ImagePreprocessingMethodEnum.gamma_correction,
                "features_idx": None,
                "step_args": {
                    "channels_idx": [1],
                    "gamma": 2.0,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [0.0**2, 0.25**2, 0.5**2, 1.0**2],
        [0.75**2, np.nan, 0.04**2, 0.81**2],
    ], dtype=np.float32)

    valid_mask = ~np.isnan(expected)
    assert np.allclose(result[:, :, 1][valid_mask], expected[valid_mask], atol=1e-6)
    assert np.isnan(result[1, 1, 1])

    assert np.allclose(result[:, :, 0], X[:, :, 0], atol=1e-6)


def test_preprocessing_log_transform():
    X = np.array([
        [
            [1, 0.0],
            [2, 1.0],
            [3, 3.0],
            [4, 7.0],
        ],
        [
            [5, 15.0],
            [6, np.nan],
            [7, 31.0],
            [8, 63.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.image_preprocessing: [{
                "method": ImagePreprocessingMethodEnum.log_transformation,
                "features_idx": None,
                "step_args": {
                    "channels_idx": [1],
                    "eps": 1e-6,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [np.log(0.0 + 1e-6), np.log(1.0 + 1e-6), np.log(3.0 + 1e-6), np.log(7.0 + 1e-6)],
        [np.log(15.0 + 1e-6), np.nan, np.log(31.0 + 1e-6), np.log(63.0 + 1e-6)],
    ], dtype=np.float32)

    valid_mask = ~np.isnan(expected)
    assert np.allclose(result[:, :, 1][valid_mask], expected[valid_mask], atol=1e-6)
    assert np.isnan(result[1, 1, 1])

    assert np.allclose(result[:, :, 0], X[:, :, 0], atol=1e-6)


def test_preprocessing_mean_imputation():
    X = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, np.nan, 2.],
            [np.nan, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [np.nan, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_mean,
                "features_idx": [1, 2],
                "step_args": None,
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, 55.0, 2.],
            [7.5, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [3.0, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6)


def test_preprocessing_median_imputation():
    X = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, np.nan, 2.],
            [np.nan, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [np.nan, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_median,
                "features_idx": [1, 2],
                "step_args": None,
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, 55.0, 2.],
            [7.5, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [3.0, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6)


def test_preprocessing_constant_imputation():
    X = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, np.nan, 2.],
            [np.nan, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [np.nan, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_constant,
                "features_idx": [1, 2],
                "step_args": {
                    "constant": -1.0,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.],
            [2.0, -1.0, 2.],
            [-1.0, 30.0, 3],
            [4.0, 9.0, 4],
        ],
        [
            [5.0, 50.0, 4.],
            [-1.0, 60.0, 4.],
            [7.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ],
        [
            [5.0, 50.0, 4.],
            [4.0, 50.0, 4.],
            [8.0, 8.0, 3.],
            [8.0, 80.0, 2.],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6)


def test_preprocessing_fill_imputation_forward():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [np.nan, 100.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, np.nan, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, np.nan, 2.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_fill,
                "features_idx": [1, 2, 3],
                "step_args": {
                    "direction": "forward",
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [np.nan, 100.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, 30.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, 80.0, 2.0],
        ],
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6, equal_nan=True)


def test_preprocessing_fill_imputation_backward():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [np.nan, 100.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, np.nan, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, np.nan, 2.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_fill,
                "features_idx": [1, 2],
                "step_args": {
                    "direction": "backward",
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [6.0, 100.0, 2.0],
            [7.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, np.nan, 2.0],
        ],
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6, equal_nan=True)


def test_preprocessing_rolling_imputation_mean_center():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, np.nan, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [np.nan, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [4.0, 50.0, 4.0],
            [8.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_rolling,
                "features_idx": [1, 2],
                "step_args": {
                    "window_size": 3,
                    "method": "mean",
                    "center": True,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 60.0, 2.0],
            [7.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [3.0, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [4.0, 50.0, 4.0],
            [8.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6)


def test_preprocessing_rolling_imputation_median_backward_window():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, np.nan, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [np.nan, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [4.0, 50.0, 4.0],
            [8.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_rolling,
                "features_idx": [1, 2],
                "step_args": {
                    "window_size": 2,
                    "method": "median",
                    "center": False,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, np.nan, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [2.0, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [4.0, 50.0, 4.0],
            [8.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6, equal_nan=True)


def test_preprocessing_kalman_imputation():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, np.nan, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [np.nan, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [4.0, 50.0, 4.0],
            [8.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_kalman,
                "features_idx": [1, 2],
                "step_args": None,
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    assert not np.isnan(result[0, 1, 1])
    assert not np.isnan(result[0, 2, 0])
    assert not np.isnan(result[1, 1, 0])

    assert np.allclose(result[:, 0, :], X[:, 0, :], atol=1e-6)
    assert np.allclose(result[:, 3, :], X[:, 3, :], atol=1e-6)
    assert np.allclose(result[:, 1, 2], X[:, 1, 2], atol=1e-6)
    assert np.allclose(result[:, 2, 1], X[:, 2, 1], atol=1e-6)
    assert np.allclose(result[:, 2, 2], X[:, 2, 2], atol=1e-6)

    # Check that the imputed values are within the expected range

    assert 45.0 <= result[0, 1, 1] <= 65.0
    assert 6.0 <= result[0, 2, 0] <= 9.0
    assert 2.0 <= result[1, 1, 0] <= 4.0


def test_preprocessing_linear_interpolation():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, np.nan, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [8.0, 8.0, 3.0],
            [12.0, np.nan, 2.0],
        ]
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_linear_inter,
                "features_idx": [1, 2, 3],
                "step_args": None,
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 60.0, 2.0],
            [7.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [8.0, 8.0, 3.0],
            [12.0, 80.0, 2.0],
        ]
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6)


def test_preprocessing_polynomial_interpolation():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 0.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, np.nan, 4.0],
            [1.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 4.0, 4.0],
            [4.0, 8.0, 3.0],
            [12.0, 70.0, 2.0],
        ],
        [
            [13.0, 130.0, 4.0],
            [14.0, 9.0, 4.0],
            [9.0, 8.0, 3.0],
            [16.0, 60.0, 2.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_polynomial_inter,
                "features_idx": [1, 2],
                "step_args": {
                    "degree": 2,
                    "window_size": 64,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 0.0, 2.0],
            [0.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 1.0, 4.0],
            [1.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 4.0, 4.0],
            [4.0, 8.0, 3.0],
            [12.0, 70.0, 2.0],
        ],
        [
            [13.0, 130.0, 4.0],
            [14.0, 9.0, 4.0],
            [9.0, 8.0, 3.0],
            [16.0, 60.0, 2.0],
        ],
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-5)


def test_preprocessing_spline_interpolation():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 0.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, np.nan, 4.0],
            [1.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 4.0, 4.0],
            [4.0, 8.0, 3.0],
            [12.0, 70.0, 2.0],
        ],
        [
            [13.0, 130.0, 4.0],
            [14.0, 9.0, 4.0],
            [9.0, 8.0, 3.0],
            [16.0, 60.0, 2.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_spline_inter,
                "features_idx": [1, 2],
                "step_args": {
                    "window_size": 64,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [2.0, 0.0, 2.0],
            [0.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 1.0, 4.0],
            [1.0, 8.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 4.0, 4.0],
            [4.0, 8.0, 3.0],
            [12.0, 70.0, 2.0],
        ],
        [
            [13.0, 130.0, 4.0],
            [14.0, 9.0, 4.0],
            [9.0, 8.0, 3.0],
            [16.0, 60.0, 2.0],
        ],
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-5)


def test_multiple_imputation_step():
    X = np.array([
        [
            [1.0, 10.0, 1.0],
            [np.nan, 100.0, 2.0],
            [np.nan, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, np.nan, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, np.nan, 2.0],
        ],
    ], dtype=np.float32)

    td = TensorData.create(X, backend_name="cpu", data_type="time_series")

    service = TSPreprocessingService()
    preprocessed_data = service.fit_transform(
        td,
        {
            PreprocessingStepEnum.imputation: [{
                "method": ImputationMethodEnum.ts_fill,
                "features_idx": [1, 2, 3],
                "step_args": {
                    "direction": "forward",
                },
            },
            {
                "method": ImputationMethodEnum.ts_constant,
                "features_idx": [1, 2, 3],
                "step_args": {
                    "constant": -1.0,
                },
            }]
        }
    )

    result = preprocessed_data.features.numpy()

    expected = np.array([
        [
            [1.0, 10.0, 1.0],
            [-1.0, 100.0, 2.0],
            [-1.0, 30.0, 3.0],
            [4.0, 9.0, 4.0],
        ],
        [
            [5.0, 50.0, 4.0],
            [6.0, 60.0, 4.0],
            [7.0, 30.0, 3.0],
            [8.0, 80.0, 2.0],
        ],
        [
            [9.0, 90.0, 4.0],
            [10.0, 110.0, 4.0],
            [11.0, 8.0, 3.0],
            [12.0, 80.0, 2.0],
        ],
    ], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-6, equal_nan=True)
