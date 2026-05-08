import numpy as np
import pandas as pd
import pytest
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.reader.ucr_loader import TSLoader
from fedot.core.data.tensor_data.lazy_tensor import LazyTensor
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.utils import fedot_project_root

# Test loading from different sources and data types
# --------------------------------------------------


@pytest.mark.unit
def test_create_from_numpy():
    """
    Test creation of TensorData from a NumPy array, ensuring that features and target
    are correctly separated and converted into torch tensors with expected shapes.
    """

    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(
        features,
        backend_name="cpu"
    )
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1


@pytest.mark.integration
def test_create_from_csv():
    """
    Test creation of TensorData from a CSV file, verifying that data is properly loaded,
    the specified target column is extracted, and both features and target are converted
    into torch tensors.
    """

    csv_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'

    td = TensorDataCreator.create(
        csv_path,
        backend_name="cpu",
        target_idx="target"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)


@pytest.mark.unit
def test_from_tensor():
    """
    Test creation of TensorData from a torch tensor, ensuring that features are preserved,
    no target is assigned, and the tensor is moved to the expected device with correct shape.
    """
    features = torch.rand(100, 10)

    td = TensorDataCreator.create(features, backend_name="cpu",)

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cpu"


@pytest.mark.integration
def test_loader():
    """
    Test TensorData creation using UCR dataset loaded via TSLoader, ensuring that both training
    and test datasets are correctly converted into TensorData objects with valid feature
    and target tensors.
    """
    name = "AbnormalHeartbeat"
    X_train, y_train, X_test, y_test = TSLoader.download_by_url(dataset_name=name)

    train_tensor = TensorDataCreator.create(X_train,
                                            target=y_train,
                                            backend_name="cpu",)
    test_tensor = TensorDataCreator.create(X_test,
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
@pytest.mark.unit
def test_create_lazy_does_not_materialize_immediately():
    """
    Test that lazy TensorData creation does not immediately materialize the underlying data,
    and that calling get() properly constructs a TensorData instance with expected features.
    """
    X = np.random.rand(10, 3)

    lazy_td = TensorDataCreator.create_lazy(
        X,
        backend_name="cpu",
    )

    assert isinstance(lazy_td, LazyTensor)
    assert lazy_td._data is None

    td = lazy_td.get()

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape == (10, 2)
# --------------------------------------------------


# Test some cases with types, nans. Detecting target. Testing counting memory.
# --------------------------------------------------
@pytest.mark.unit
def test_datetime_features():
    """
    Test that datetime columns are handled correctly during TensorData creation,
    with the target column excluded from features and all data converted into valid tensors.
    """
    features = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=10, freq="D"),
                             "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    td = TensorDataCreator.create(features, backend_name="cpu", target_idx="target")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target.shape[0] == features.shape[0]
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1


@pytest.mark.unit
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

    td = TensorDataCreator.create(
        X,
        target=y,
        backend_name="cpu"
    )

    assert td.features.shape[0] == 2
    assert td.target.shape[0] == 2


@pytest.mark.unit
def test_target_extracted_by_index():
    """
    Test that the target column can be extracted by its positional index,
    with the remaining columns correctly used as feature data.
    """
    X = np.random.rand(20, 5)

    td = TensorDataCreator.create(
        X,
        target_idx=2,
        backend_name="cpu"
    )

    assert isinstance(td.target, torch.Tensor)
    assert td.features.shape[1] == 4
    assert td.target.shape[0] == 20


@pytest.mark.unit
def test_create_from_numpy_without_target():
    """
    Test creation of TensorData from a NumPy array without target, ensuring that features
    are correctly separated and converted into torch tensors with expected shapes.
    """

    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(
        features,
        backend_name="cpu",
        without_target=True
    )
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target is None
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1]


# # TODO romankuklo: state='PREDICT' is not implemented yet
# # def test_target_depends_state():
# #     """
# #     Test that target handling depends on the processing state, ensuring that in predict mode
# #     no target is inferred or created and all input columns are treated as features.
# #     """
# #     features = torch.rand(100, 5)

# #     td = TensorData.create(features, backend_name="cpu", state="predict")

# #     assert isinstance(td, TensorData)
# #     assert isinstance(td.features, torch.Tensor)
# #     assert td.target is None
# #     assert td.features.shape[0] == features.shape[0]
# #     assert td.features.shape[1] == features.shape[1]


@pytest.mark.unit
def test_memory_count_cpu():
    """
    Test that the memory count of a TensorData instance is correctly calculated,
    including the size of the features tensor and the target tensor if present.
    """
    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(features, backend_name="cpu")

    assert td.features.device.type == "cpu"
    assert td.memory_usage["features"] > 0
    assert td.memory_usage["target"] >= 0
    assert td.memory_usage["predict"] == 0
    assert td.memory_usage["metadata"] > 0
    assert td.memory_usage["total"] == (
        td.memory_usage["features"]
        + td.memory_usage["target"]
        + td.memory_usage["predict"]
        + td.memory_usage["metadata"]
    )
# --------------------------------------------------


# Test Time Series preprocessing
# --------------------------------------------------
@pytest.mark.unit
def test_create_time_series():
    """
    Test creation of TensorData from a time series dataset, ensuring that features and target
    are correctly separated and converted into torch tensors with expected shapes.
    """
    Backend().set("cpu")
    features = np.random.rand(100, 10)

    td = TensorDataCreator.create(
        features,
        backend_name="cpu",
        data_type="time_series",
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1]


@pytest.mark.unit
def test_long_orientation():
    """
    Test that TensorData supports long orientation, ensuring that features
    are correctly separated and converted into torch tensors with expected shapes.
    """
    X = pd.DataFrame({
        'terms': ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        'vals': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })

    td = TensorDataCreator.create(
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


@pytest.mark.unit
def test_is_multichannel():
    """
    Test that TensorData supports multi-channel data, ensuring that features save the correct shape.
    """
    X = np.random.rand(100, 10, 3)

    td = TensorDataCreator.create(
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
