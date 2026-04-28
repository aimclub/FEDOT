import numpy as np
import torch
import cupy as cp
import cudf
import pandas as pd

from fedot.core.data.tensordata import TensorData, LazyTensor
from fedot.core.data.ucr_loader import TSLoader
from fedot.core.utils import fedot_project_root
from fedot.core.backend.backend import Backend


# Test loading from different sources and data types
# --------------------------------------------------
def test_create_from_numpy():
    """
    Test creation of TensorData from a NumPy array, ensuring that features and target
    are correctly separated and converted into torch tensors with expected shapes.
    """

    features = np.random.rand(100, 10)

    td = TensorData.create(
        features,
        backend_name="cpu"
    )
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1


def test_create_from_csv():
    """
    Test creation of TensorData from a CSV file, verifying that data is properly loaded,
    the specified target column is extracted, and both features and target are converted
    into torch tensors.
    """

    csv_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'

    td = TensorData.create(
        csv_path,
        backend_name="cpu",
        target_idx = "target"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)


def test_from_tensor():
    """
    Test creation of TensorData from a torch tensor, ensuring that features are preserved,
    no target is assigned, and the tensor is moved to the expected device with correct shape.
    """
    features = torch.rand(100, 10)

    td = TensorData.create(features, backend_name="cpu",)

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cpu"


def test_loader():
    """
    Test TensorData creation using UCR dataset loaded via TSLoader, ensuring that both training
    and test datasets are correctly converted into TensorData objects with valid feature
    and target tensors.
    """
    name = "AbnormalHeartbeat"
    X_train, y_train, X_test, y_test = TSLoader().download_by_url(dataset_name=name)

    train_tensor = TensorData.create(X_train, 
                                     target=y_train,
                                     backend_name="cpu",)
    test_tensor = TensorData.create(X_test, 
                                    target=y_test,
                                    backend_name="cpu",)

    assert isinstance(train_tensor, TensorData)
    assert isinstance(test_tensor, TensorData)

    assert isinstance(train_tensor.features, torch.Tensor)
    assert isinstance(train_tensor.target, torch.Tensor)
    assert isinstance(test_tensor.features, torch.Tensor)
    assert isinstance(test_tensor.target, torch.Tensor)


# Test lazy tensordata
# --------------------------------------------------
def test_create_lazy_does_not_materialize_immediately():
    """
    Test that lazy TensorData creation does not immediately materialize the underlying data,
    and that calling get() properly constructs a TensorData instance with expected features.
    """
    X = np.random.rand(10, 3)

    lazy_td = TensorData.create_lazy(
        X,
        backend_name="cpu",
    )

    assert isinstance(lazy_td, LazyTensor)
    assert lazy_td._data is None

    td = lazy_td.get()

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape == (10, 2)


def test_lazy_tensordata_to_device():
    """
    Test that a LazyTensor can be materialized directly onto a specified device using to(),
    ensuring that the resulting TensorData has its features tensor on the correct device.
    """
    X = np.random.rand(4, 3)

    lazy_td = TensorData.create_lazy(
        X,
        backend_name="gpu",
    )

    td = lazy_td.to("cpu")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cpu"
# --------------------------------------------------


# Test some cases with types, nans. Detecting target. Testing counting memory.
# --------------------------------------------------
def test_datetime_features():
    """
    Test that datetime columns are handled correctly during TensorData creation,
    with the target column excluded from features and all data converted into valid tensors.
    """
    features = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=10, freq="D"), 
                             "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    td = TensorData.create(features, backend_name="cpu", target_idx="target")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1


def test_nan_rows_are_dropped_from_target():
    """
    Test that rows with missing target values are removed during TensorData creation,
    ensuring that features and target remain aligned after dropping invalid samples.
    """
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y = np.array([1.0, None, 0.0], dtype=object)

    td = TensorData.create(
        X,
        target=y,
        backend_name="cpu"
    )

    assert td.features.shape[0] == 2
    assert td.target.shape[0] == 2


def test_target_extracted_by_index():
    """
    Test that the target column can be extracted by its positional index,
    with the remaining columns correctly used as feature data.
    """
    X = np.random.rand(20, 5)

    td = TensorData.create(
        X,
        target_idx=2,
        backend_name="cpu"
    )

    assert isinstance(td.target, torch.Tensor)
    assert td.features.shape[1] == 4
    assert td.target.shape[0] == 20


# TODO romankuklo: state='PREDICT' is not implemented yet
# def test_target_depends_state():
#     """
#     Test that target handling depends on the processing state, ensuring that in predict mode
#     no target is inferred or created and all input columns are treated as features.
#     """
#     features = torch.rand(100, 5)

#     td = TensorData.create(features, backend_name="cpu", state="predict")

#     assert isinstance(td, TensorData)
#     assert isinstance(td.features, torch.Tensor)
#     assert td.target is None
#     assert td.features.shape[0] == features.shape[0]
#     assert td.features.shape[1] == features.shape[1]


def test_memory_count_cpu():
    """
    Test that the memory count of a TensorData instance is correctly calculated,
    including the size of the features tensor and the target tensor if present.
    """
    features = np.random.rand(100, 10)

    td = TensorData.create(features, backend_name="cpu")

    assert td.features.device.type == "cpu"
    assert td.memory_usage > 0


def test_memory_count_gpu():
    """
    Test that the memory count of a TensorData instance on the GPU is correctly calculated,
    including the size of the features tensor and the target tensor if present.
    """
    Backend().set("gpu")
    
    features = np.random.rand(100, 10)

    td = TensorData.create(features, backend_name="gpu")

    assert td.features.device.type == "cuda"
    assert td.memory_usage > 0

# --------------------------------------------------


# Test GPU backend
# --------------------------------------------------
def test_create_from_numpy_cupy():
    """
    Test TensorData creation from a NumPy array using the GPU backend, ensuring that data
    is correctly transferred to CUDA tensors and feature-target separation is preserved.
    """

    Backend().set("gpu")
    features = np.random.rand(100, 10)

    td = TensorData.create(
        features,
        backend_name="gpu"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cuda"
    assert td.target.device.type == "cuda"


def test_create_from_cupy():
    """
    Test TensorData creation from a CuPy array, verifying that GPU-native data is properly
    converted into torch tensors on the CUDA device with correct shapes.
    """
    features = cp.random.rand(100, 10)

    td = TensorData.create(
        features,
        backend_name="gpu",
    )
    
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cuda"
    assert td.target.device.type == "cuda"


def test_create_from_cudf():
    """
    Test TensorData creation from a cuDF DataFrame, ensuring that GPU DataFrame inputs
    are correctly processed into CUDA tensors with proper feature-target handling.
    """
    features = cudf.DataFrame(np.random.rand(100, 10))

    td = TensorData.create(
        features,
        backend_name="gpu",
    )
    
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cuda"
    assert td.target.device.type == "cuda"


def test_nan_gpu_backend():
    x = np.array([[1, np.nan, 3], [4, None, 6]])
    td = TensorData.create(x, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cuda"


def test_categorical_encoding_gpu_backend():
    X = np.array([
        [100, "A", 10],
        [500, "B", 20],
        [100, "A", 30],
    ])

    td = TensorData.create(X, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert td.features.device.type == "cuda"

# --------------------------------------------------


# Test Time Series preprocessing
# --------------------------------------------------
def test_create_time_series():
    """
    Test creation of TensorData from a time series dataset, ensuring that features and target
    are correctly separated and converted into torch tensors with expected shapes.
    """
    Backend().set("cpu")
    features = np.random.rand(100, 10)

    td = TensorData.create(
        features,
        backend_name="cpu",
        data_type="time_series",
    )
    
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1]


def test_long_orientation():
    """
    Test that TensorData supports long orientation, ensuring that features
    are correctly separated and converted into torch tensors with expected shapes.
    """
    X = pd.DataFrame({
        'terms': ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        'vals': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })

    td = TensorData.create(
        X,
        backend_name="cpu",
        data_type="time_series",
        ts_orientation="long",
        ts_terms_idx="terms"
    )
    
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == 3
    assert td.features.shape[1] == 3


def test_is_multichannel():
    """
    Test that TensorData supports multi-channel data, ensuring that features save the correct shape.
    """
    X = np.random.rand(100, 10, 3)

    td = TensorData.create(
        X, 
        backend_name="cpu",
        data_type="time_series"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target is None
    assert td.features.shape[0] == X.shape[0]
    assert td.features.shape[1] == X.shape[1]
    assert td.features.shape[2] == X.shape[2]
# --------------------------------------------------
