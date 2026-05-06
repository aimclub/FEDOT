import pytest

cp = pytest.importorskip("cupy")
cudf = pytest.importorskip("cudf")

from fedot.core.data.tensor_data.td_creator import TensorDataCreator
from fedot.core.data.tensor_data.tensor_data import TensorData
import numpy as np
import torch

from fedot.core.backend.backend import Backend


# Test lazy tensordata
# --------------------------------------------------
def test_lazy_tensordata_to_device():
    """
    Test that a LazyTensor can be materialized directly onto a specified device using to(),
    ensuring that the resulting TensorData has its features tensor on the correct device.
    """
    X = np.random.rand(4, 3)

    lazy_td = TensorDataCreator.create_lazy(
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
def test_memory_count_gpu():
    """
    Test that the memory count of a TensorData instance on the GPU is correctly calculated,
    including the size of the features tensor and the target tensor if present.
    """
    Backend().set("gpu")

    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(features, backend_name="gpu")

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

    td = TensorDataCreator.create(
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

    td = TensorDataCreator.create(
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

    td = TensorDataCreator.create(
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
    td = TensorDataCreator.create(x, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cuda"


def test_categorical_encoding_gpu_backend():
    X = np.array([
        [100, "A", 10],
        [500, "B", 20],
        [100, "A", 30],
    ])

    td = TensorDataCreator.create(X, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert td.features.device.type == "cuda"
# --------------------------------------------------
