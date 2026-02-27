import numpy as np
import torch
import os
from fedot.core.data.tensordata import TensorData, LazyTensor
from fedot.core.utils import fedot_project_root


def test_create_from_numpy():
    """Test TensorData creation from numpy array."""

    features = np.random.rand(100, 10)

    td = TensorData.create(
        features
    )
    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == (features.shape[1] - 1)


def test_create_from_csv():
    """Test TensorData creation from CSV file."""

    csv_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'

    td = TensorData.create(
        csv_path
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)


def test_nan_rows_are_dropped_from_target():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y = np.array([1.0, None, 0.0], dtype=object)

    td = TensorData.create(
        X,
        target=y
    )

    assert td.features.shape[0] == 2
    assert td.target.shape[0] == 2


def test_target_extracted_by_index():
    X = np.random.rand(20, 5)

    td = TensorData.create(
        X,
        target_idx=2
    )

    assert isinstance(td.target, torch.Tensor)
    assert td.features.shape[1] == 4
    assert td.target.shape[0] == 20


def test_text_and_categorical_do_not_overlap():
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
        categorical_idx=np.array([1])
    )

    assert td.categorical_idx is not None
    assert isinstance(td.categorical_features, torch.Tensor)
    assert td.categorical_features.shape[1] == 1


def test_categorical_features_match_indices():
    X = np.array([
        [1, "A", 10],
        [2, "B", 20],
        [3, "A", 30],
    ], dtype=object)
    columns = ["id", "text", "number"]
    td = TensorData.create(
        X,
        categorical_idx=np.array(["text"]),
        features_names=columns
    )

    assert td.categorical_features is not None
    assert td.categorical_features.shape[1] == len(td.categorical_idx)


def test_create_text_csv_to_tensordata():
    """Test text CSV → TensorData with text embeddings."""

    csv_path = (
        f"{fedot_project_root()}/examples/real_cases/data/spam/spamham.csv"
    )

    assert os.path.exists(csv_path)

    td = TensorData.create(
        csv_path,
        max_rows=10,
        embedder_batch_size=3        
    )

    assert isinstance(td, TensorData)

    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)

    assert td.features.ndim == 2
    assert td.target.ndim in (1, 2)

    assert td.features.shape[0] == 10
    assert td.target.shape[0] == 10

    assert td.features.shape[1] > 1


def test_create_lazy_does_not_materialize_immediately():
    X = np.random.rand(10, 3)

    lazy_td = TensorData.create_lazy(
        X,
    )

    assert isinstance(lazy_td, LazyTensor)
    assert lazy_td._data is None

    td = lazy_td.get()

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape == (10, 2)


def test_lazy_tensordata_to_device():
    X = np.random.rand(4, 3)

    lazy_td = TensorData.create_lazy(
        X,
    )

    td = lazy_td.to("cpu")

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.device.type == "cpu"
