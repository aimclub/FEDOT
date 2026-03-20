import numpy as np
import torch
import os
from fedot.core.data.tensordata import TensorData, LazyTensor
from fedot.core.data.ucr_loader import TSLoader
from fedot.core.utils import fedot_project_root
from fedot.core.backend.backend import backend

import cupy as cp
import cudf

import pandas as pd


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
    assert td.target is None
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1]
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
# --------------------------------------------------


# Test encoding and getting embeddings
# --------------------------------------------------
def test_text_features_preprocess_like_categorical():
    """
    Test that text features are detected and encoded like categorical,
    if no embedding strategy is provided.
    """
    X = np.array([
        ["date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg", 
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 1],
        ["martin a posted tassos papadopoulos the greek sculptor behind", 
         "i just had to jump in here as carbonara is one of my favourites to make", 2],
        ["man threatens explosion in moscow thursday august NUMBER NUMBER NUMBER NUMBER pm", 
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 3],
    ], dtype=object)

    td = TensorData.create(
        X,
        backend_name="cpu"
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == 2


def test_categorical_features_match_indices():
    """
    Test that categorical features are correctly matched to the provided column indices or names
    and are properly encoded into the resulting TensorData feature tensor.
    """
    X = np.array([
        [1, "A", 10],
        [2, "B", 20],
        [3, "A", 30],
    ], dtype=object)
    columns = ["id", "text", "number"]
    td = TensorData.create(
        X,
        backend_name="cpu",
        categorical_idx=np.array(["text"]),
        features_names=columns
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features is not None
    assert td.features.shape[1] == 2


def test_create_text_csv_to_tensordata():
    """
    Test creation of TensorData from a CSV file containing text data, verifying that text features
    are converted into embeddings and that both features and target tensors have the expected shapes.
    """

    csv_path = (
        f"{fedot_project_root()}/examples/real_cases/data/spam/spamham.csv"
    )

    assert os.path.exists(csv_path)

    td = TensorData.create(
        csv_path,
        backend_name="gpu",
        text_idx = 0,
        embedding_strategy={
            "method": "sentence_transformer",
            "model_name": "all-distilroberta-v1",
            "batch_size": 3,
            "device": "cuda"
        },
        max_rows=10,    
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)

    assert td.features.ndim == 2

    assert td.features.shape[0] == 10
    assert td.target.shape[0] == 10

    assert td.features.shape[1] > 1


def test_categorical_text():
    """
    Test combined processing of text, numerical, and categorical features, ensuring that text columns
    are embedded, categorical columns are encoded, and all transformed features are concatenated correctly.
    """
    X = np.array([
        ["date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg", 
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 1, "A", "DOP", 0],
        ["martin a posted tassos papadopoulos the greek sculptor behind", 
         "i just had to jump in here as carbonara is one of my favourites to make", 1, "A", "DROP", 0],
        ["man threatens explosion in moscow thursday august NUMBER NUMBER NUMBER NUMBER pm", 
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 3, "B", "DOP", 1],
    ], dtype=object)

    columns = ["text1", "text2", "number", "class", "subclass", "target"]

    encoding_strategy = {
        "label": ["class", "subclass"]
    }

    embedding_strategy= {
            "method": "sentence_transformer",
            "model_name": "all-distilroberta-v1",
            "batch_size": 3,
            "device": "cpu",
    }
    text_idx = ["text1", "text2"]

    td = TensorData.create(
        X,
        backend_name="cpu",
        features_names=columns,
        encoding_strategy=encoding_strategy,
        text_idx=text_idx,
        embedding_strategy=embedding_strategy
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.ndim == 2
    assert td.features.shape[1] == 768 * 2 + 3


def test_label_ohe_encoding():
    """
    Test application of mixed label encoding and one-hot encoding strategies, verifying that categorical
    features are transformed correctly and the resulting feature tensor has the expected dimensionality.
    """
    X = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "D", 30],
    ])

    encoding_strategy = {
        "label": [0, 1],
        "ohe": [2]
    }

    td = TensorData.create(
        X,
        backend_name="cpu",
        encoding_strategy=encoding_strategy,
        target_idx=3
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == 4


def test_save_encoding_after_fit():
    """
    Test that fitted encoding settings are preserved and reused during prediction, ensuring that test data
    is transformed consistently with training data and produces identical encoded features when applicable.
    """
    X_train = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "D", 30],
    ])

    X_test = np.array([
        [100, "A", "C"],
        [500, "B", "D"],
        [100, "A", "D"],
    ])

    td_train = TensorData.create(
        X_train,
        backend_name="cpu",
        target_idx=3
    )
    
    td_test = TensorData.create(
        X_test,
        backend_name="cpu",
        state="predict",
        encoding_strategy=td_train.encoding_strategy,
    )

    assert isinstance(td_test.features, torch.Tensor)
    assert td_test.features.shape[1] == td_train.features.shape[1]
    assert torch.equal(td_test.features, td_train.features)

def test_encoding_torch_data():
    """
    Test encoding of torch data, ensuring that categorical features are transformed correctly
    and the resulting feature tensor has the expected dimensionality.
    """
    X = torch.rand(10, 4)

    encoding_strategy = {
        "label": [0, 1],
    }

    td = TensorData.create(
        X,
        backend_name="cpu",
        encoding_strategy=encoding_strategy,
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == 4
# --------------------------------------------------


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
        backend_name="cpu",
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


def test_target_depends_state():
    """
    Test that target handling depends on the processing state, ensuring that in predict mode
    no target is inferred or created and all input columns are treated as features.
    """
    features = torch.rand(100, 5)

    td = TensorData.create(features, backend_name="cpu", state="predict")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.target is None
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1]


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
    backend.set("gpu")
    
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

    backend.set("gpu")
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


def test_create_from_csv_gpu():
    """
    Test creation of TensorData from a CSV file, verifying that data is properly loaded,
    the specified target column is extracted, and both features and target are converted
    into torch tensors.
    """

    csv_path = f'{fedot_project_root()}/examples/data/credit_card_anomaly.csv'

    td = TensorData.create(
        csv_path,
        backend_name="gpu",
        target_idx = "Class"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)
    assert td.features.device.type == "cuda"


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
    backend.set("cpu")
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
    assert td.features.shape[0] == X.shape[0]
    assert td.features.shape[1] == X.shape[1]
    assert td.features.shape[2] == X.shape[2]
# --------------------------------------------------
