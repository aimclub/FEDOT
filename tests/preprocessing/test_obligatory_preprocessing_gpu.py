from fedot.preprocessing.tools.preprocessor_types import EmbeddingMethodEnum
from fedot.core.utils import fedot_project_root
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.backend.backend import Backend
import os

import numpy as np
import pytest
import torch

cp = pytest.importorskip("cupy", exc_type=ImportError)
cudf = pytest.importorskip("cudf", exc_type=ImportError)

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for GPU TensorData tests", allow_module_level=True)


@pytest.mark.integration
def test_create_text_csv_to_tensordata():
    """Test CSV text data conversion with transformer embeddings.

    Checks that spam text data is loaded from CSV, limited to 10 rows, converted to
    feature/target tensors, and expanded into a 2D embedding feature matrix."""

    csv_path = (
        f"{fedot_project_root()}/examples/real_cases/data/spam/spamham.csv"
    )

    assert os.path.exists(csv_path)

    embedding_strategy = [{
        "method": EmbeddingMethodEnum.transformer,
        "model_name": "all-distilroberta-v1",
        "batch_size": 3,
        "device": torch.device("cuda"),
        "features_idx": [0]
    }]

    td = TensorDataCreator.create(
        csv_path,
        backend_name="gpu",
        embedding_strategy=embedding_strategy,
        max_rows=10,
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)

    assert td.features.ndim == 2

    assert td.features.shape[0] == 10
    assert td.target.shape[0] == 10

    assert td.features.shape[1] > 1


@pytest.mark.unit
def test_lazy_tensordata_to_device():
    """Test lazy TensorData materialization to a requested device.

    Checks that a GPU-configured LazyTensor can be materialized via `to("cpu")` and
    that resulting feature tensor is on CPU."""
    X = np.random.rand(4, 3)

    lazy_td = TensorDataCreator.create_lazy(
        X,
        backend_name="gpu",
    )

    td = lazy_td.to("cpu")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cpu"


@pytest.mark.unit
def test_create_from_numpy_cupy():
    """Test NumPy input conversion with GPU backend.

    Checks that NumPy data is converted into CUDA feature/target tensors and default
    last-column target extraction preserves sample count."""

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


@pytest.mark.unit
def test_create_from_cupy():
    """Test CuPy input conversion to TensorData on GPU.

    Checks that GPU-native CuPy data becomes CUDA torch tensors with expected
    feature/target shapes after default target extraction."""
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


@pytest.mark.unit
def test_create_from_cudf():
    """Test cuDF DataFrame input conversion on GPU.

    Checks that cuDF data is processed into CUDA feature and target tensors with the
    same row count and one fewer feature column."""
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


@pytest.mark.integration
def test_create_from_csv_gpu():
    """Test CSV loading with GPU backend and named target.

    Checks that credit card anomaly CSV data is loaded, `Class` is used as target,
    and features are placed on CUDA."""

    csv_path = f'{fedot_project_root()}/examples/data/credit_card_anomaly.csv'

    td = TensorDataCreator.create(
        csv_path,
        backend_name="gpu",
        target_idx="Class"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)
    assert td.features.device.type == "cuda"


@pytest.mark.unit
def test_nan_gpu_backend():
    """Test GPU TensorData creation with missing values.

    Checks that arrays containing `np.nan` and `None` are accepted and converted to a
    CUDA feature tensor."""
    x = np.array([[1, np.nan, 3], [4, None, 6]])
    td = TensorDataCreator.create(x, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cuda"


@pytest.mark.unit
def test_categorical_encoding_gpu_backend():
    """Test automatic categorical encoding on GPU backend.

    Checks that object categorical values are converted during TensorData creation
    and final features reside on CUDA."""
    X = np.array([
        [100, "A", 10],
        [500, "B", 20],
        [100, "A", 30],
    ])

    td = TensorDataCreator.create(X, backend_name="gpu")
    assert isinstance(td, TensorData)
    assert td.features.device.type == "cuda"
