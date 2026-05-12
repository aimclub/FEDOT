from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.data.reader import TSLoader
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum, EmbeddingMethodEnum


@pytest.mark.unit
def test_text_features_preprocess_like_categorical():
    """Test automatic text handling without embedding strategy.

    Checks that object text columns are treated like categorical features and that
    the resulting tensor keeps two feature columns after target separation."""
    X = np.array([
        ["date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg",
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 1],
        ["martin a posted tassos papadopoulos the greek sculptor behind",
         "i just had to jump in here as carbonara is one of my favourites to make", 2],
        ["man threatens explosion in moscow thursday august NUMBER NUMBER NUMBER NUMBER pm",
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 3],
    ], dtype=object)

    td = TensorDataCreator.create(
        X,
        backend_name="cpu"
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == 2


@pytest.mark.unit
def test_categorical_features_match_indices():
    """Test categorical column selection by feature name.

    Checks that the named categorical column is resolved through `features_names`,
    encoded, and the output feature tensor has the expected reduced width."""
    X = np.array([
        [1, "A", 10],
        [2, "B", 20],
        [3, "A", 30],
    ], dtype=object)
    columns = ["id", "text", "number"]

    strategy = [{
        "method": EncodingMethodEnum.label,
        "features_idx": ["text"]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        features_names=columns,
        encoding_strategy=strategy,
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features is not None
    assert td.features.shape[1] == 2


@pytest.mark.unit
def test_from_tensor():
    """Test TensorData creation from an input torch tensor.

    Checks default target extraction from the last column, feature shape reduction by
    one column, and placement on the requested CPU device."""
    features = torch.rand(100, 10)

    td = TensorDataCreator.create(features, backend_name="cpu",)

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert isinstance(td.target, torch.Tensor)
    assert td.features.shape[0] == features.shape[0]
    assert td.features.shape[1] == features.shape[1] - 1
    assert td.features.device.type == "cpu"


# TODO romankuklo: add after cache is implemented
# def test_loader():
#     """Test TensorData conversion for a dataset loaded by `TSLoader`.

#     Checks that both train and test parts of the UCR dataset are converted into
#     TensorData objects with torch feature and target tensors.
#     Also checks that target is encoded automatically."""
#     name = "AbnormalHeartbeat"
#     X_train, y_train, X_test, y_test = TSLoader().download_by_url(dataset_name=name)

#     train_tensor = TensorDataCreator.create(X_train,
#                                      target=y_train,
#                                      backend_name="cpu",)
#     test_tensor = TensorDataCreator.create(X_test,
#                                     target=y_test,
#                                     backend_name="cpu",)

#     assert isinstance(train_tensor, TensorData)
#     assert isinstance(test_tensor, TensorData)

#     assert isinstance(train_tensor.features, torch.Tensor)
#     assert isinstance(train_tensor.target, torch.Tensor)
#     assert isinstance(test_tensor.features, torch.Tensor)
#     assert isinstance(test_tensor.target, torch.Tensor)


@pytest.mark.integration
def test_categorical_text():
    """Test combined text embedding and categorical encoding.

    Checks that two text columns become transformer embeddings, categorical columns
    are label-encoded, and the final feature width equals two embedding blocks plus
    three remaining numeric/encoded columns."""
    X = np.array([
        ["date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg",
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 1, "A", "DOP", 0],
        ["martin a posted tassos papadopoulos the greek sculptor behind",
         "i just had to jump in here as carbonara is one of my favourites to make", 1, "A", "DROP", 0],
        ["man threatens explosion in moscow thursday august NUMBER NUMBER NUMBER NUMBER pm",
         "in adding cream to spaghetti carbonara which has the same effect on pasta", 3, "B", "DOP", 1],
    ], dtype=object)

    columns = ["text1", "text2", "number", "class", "subclass", "target"]

    encoding_strategy = [{
        "method": EncodingMethodEnum.label,
        "features_idx": ["class", "subclass"]
    }]

    embedding_strategy = [{
        "method": EmbeddingMethodEnum.transformer,
        "model_name": "all-distilroberta-v1",
        "batch_size": 3,
        "device": torch.device("cpu"),
        "features_idx": ["text1", "text2"]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        features_names=columns,
        encoding_strategy=encoding_strategy,
        embedding_strategy=embedding_strategy
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.ndim == 2
    assert td.features.shape[1] == 768 * 2 + 3


@pytest.mark.unit
def test_label_ohe_encoding():
    """Test mixed label and one-hot categorical encoding.

    Checks that selected columns are encoded by different strategies and that the
    resulting feature tensor has the expected four columns after target extraction."""
    X = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "D", 30],
    ])

    encoding_strategy = [
        {"method": EncodingMethodEnum.label,
         "features_idx": [0, 1]},
        {"method": EncodingMethodEnum.ohe,
         "features_idx": [2]}
    ]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        encoding_strategy=encoding_strategy,
        target_idx=3
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == 4


@pytest.mark.unit
def test_encoding_torch_data():
    """Test encoding strategy on torch tensor input.

    Checks that TensorData accepts a torch tensor with categorical encoding config
    and still produces the expected feature width after default target split."""
    X = torch.rand(10, 4)

    encoding_strategy = [{
        "method": EncodingMethodEnum.label,
        "features_idx": [0, 1]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        encoding_strategy=encoding_strategy,
    )

    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[1] == X.shape[1] - 1


@pytest.mark.unit
def test_datetime_features():
    """Test DataFrame input with datetime feature and named target.

    Checks that datetime columns do not break TensorData creation, target is removed
    from features, and feature/target row counts remain aligned."""
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
    """Test removal of rows with missing target values.

    Checks that a `None` target entry drops the corresponding feature row so feature
    and target tensors both contain only two valid samples."""
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
    """Test target extraction by positional column index.

    Checks that column index 2 becomes target, remaining four columns stay as
    features, and target length matches the number of samples."""
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
def test_create_time_series():
    """Test time-series TensorData creation without default target split.

    Checks that `data_type="time_series"` preserves the original 2D feature width
    and returns torch features with matching sample count."""
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
    """Test long-orientation time-series reshaping.

    Checks that repeated term labels are pivoted into three logical series with
    three values each, producing a `(3, 3)` feature tensor."""
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
    """Test multichannel time-series TensorData creation.

    Checks that a 3D array keeps sample, timestep, and channel dimensions unchanged
    in the resulting feature tensor."""
    X = np.random.rand(100, 10, 3)

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        data_type="time_series"
    )

    assert isinstance(td, TensorData)
    assert isinstance(td.features, torch.Tensor)
    assert td.features.shape[0] == X.shape[0]
    assert td.features.shape[1] == X.shape[1]
    assert td.features.shape[2] == X.shape[2]


@pytest.mark.integration
def test_update_idx_emb_enc():
    """Test index mapping when embedding and one-hot encoding are combined.

    Checks that numeric columns remain in expected positions and appended one-hot
    columns for `class`/`subclass` contain the expected indicator values after a text
    embedding step changes feature layout."""
    X = np.array([
        ["date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg", 1, "A", "DOP", 0, 1],
        ["martin a posted tassos papadopoulos the greek sculptor behind", 1, "A", "DROP", 0, 1],
        ["man threatens explosion in moscow thursday august NUMBER NUMBER NUMBER NUMBER pm", 3, "B", "DOP", 1, 1],
    ], dtype=object)

    columns = ["text1", "number", "class", "subclass", "some", "target"]

    encoding_strategy = [{
        "method": EncodingMethodEnum.ohe,
        "features_idx": ["class", "subclass"]
    }]

    embedding_strategy = [{
        "method": EmbeddingMethodEnum.transformer,
        "model_name": "all-distilroberta-v1",
        "batch_size": 3,
        "device": torch.device("cpu"),
        "features_idx": ["text1"]
    }]

    td = TensorDataCreator.create(
        X,
        backend_name="cpu",
        features_names=columns,
        encoding_strategy=encoding_strategy,
        embedding_strategy=embedding_strategy
    )

    # unchanged features
    assert td.features[0, 0] == 1
    assert td.features[0, 1] == 0
    # ohe
    assert td.features[0, -4] == 1  # A
    assert td.features[0, -3] == 0  # A
    assert td.features[0, -2] == 1  # DOP
    assert td.features[0, -1] == 0  # DOP
    assert td.features[1, -4] == 1  # A
    assert td.features[1, -3] == 0  # A
    assert td.features[1, -2] == 0  # DROP
    assert td.features[1, -1] == 1


@pytest.mark.unit
def test_update_idx_enc():
    """Test index mapping with one-hot and label encoding strategies.

    Checks that named feature selectors are resolved correctly, one-hot output is
    appended in the expected order, and label-encoded values match reference rows."""
    X = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "C", 30],
    ], dtype=object)

    features_names = ["A", "B", "C", "target"]

    encoding_strategy = [{
        "method": EncodingMethodEnum.ohe,
        "features_idx": ["B"]
    },
        {
        "method": EncodingMethodEnum.label,
        "features_idx": ["C"]
    }]

    td = TensorDataCreator.create(X, backend_name="cpu",
                                  features_names=features_names,
                                  encoding_strategy=encoding_strategy)
    assert isinstance(td, TensorData)
    assert td.features.shape[1] == 4
    assert td.features[0, 0] == 100
    assert td.features[0, 1] == 0
    assert td.features[0, 2] == 1
    assert td.features[0, 3] == 0

    assert td.features[1, 0] == 500
    assert td.features[1, 1] == 1
    assert td.features[1, 2] == 0
    assert td.features[1, 3] == 1


@pytest.mark.unit
def test_custom_encoders():
    """Test custom obligatory encoders with explicit implementations.

    Checks that custom handlers fill selected named columns with `1` and `-2`, and
    that resulting features match the reference numeric matrix."""
    X = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "C", 30],
    ], dtype=object)

    features_names = ["A", "B", "C", "target"]

    class OnesFiller(AbstractPreprocessingHandler):
        def __init__(self):
            self.categorical_idx_ = None

        def fit(self, data: PreparedData, features_idx: Sequence[int]):
            self.categorical_idx_ = list(features_idx)
            return self

        def transform(self, data: PreparedData) -> PreparedData:
            xp = Backend().xp

            features = data.features

            n_rows = features.shape[0]
            n_cat = len(self.categorical_idx_)

            filled = xp.full((n_rows, n_cat), 1.0, dtype=float)

            features[:, self.categorical_idx_] = filled
            data.features = features

            return data

    class ConstantFiller(AbstractPreprocessingHandler):
        def __init__(self, constant=0.0):
            self.constant = constant

            self.categorical_idx_ = None

        def fit(self, data: PreparedData, features_idx: Sequence[int]):
            self.categorical_idx_ = list(features_idx)
            return self

        def transform(self, data: PreparedData) -> PreparedData:
            xp = Backend().xp

            features = data.features

            n_rows = features.shape[0]
            n_cat = len(self.categorical_idx_)

            filled = xp.full((n_rows, n_cat), self.constant, dtype=float)

            features[:, self.categorical_idx_] = filled
            data.features = features

            return data

    custom_strategy = [{
        "method": 'OneFiller',
        "features_idx": ["B"],
        "implementation": OnesFiller,
        "step_args": None,
    },
        {
        "method": 'ConstantFiller',
        "features_idx": ["C"],
        "implementation": ConstantFiller,
        "step_args": {"constant": -2.0},
    }]

    td = TensorDataCreator.create(X, backend_name="cpu",
                                  features_names=features_names,
                                  custom_strategy=custom_strategy)

    np_features = td.features.numpy()

    ref_X = np.array([
        [100, 1, -2],
        [500, 1, -2],
        [100, 1, -2],
    ], dtype=np.float32)

    assert isinstance(td, TensorData)
    assert np.allclose(np_features, ref_X, atol=1e-5)


@pytest.mark.unit
def test_custom_encoders_automatic_encoding():
    """Test custom encoder together with automatic categorical encoding.

    Checks that the custom handler fills column `B` with ones while remaining
    categorical column `C` is automatically label-encoded to the expected values."""
    X = np.array([
        [100, "A", "C", 10],
        [500, "B", "D", 20],
        [100, "A", "C", 30],
    ], dtype=object)

    features_names = ["A", "B", "C", "target"]

    class OnesFiller(AbstractPreprocessingHandler):
        def __init__(self):
            self.categorical_idx_ = None

        def fit(self, data: PreparedData, features_idx: Sequence[int]):
            self.categorical_idx_ = list(features_idx)
            return self

        def transform(self, data: PreparedData) -> PreparedData:
            xp = Backend().xp

            features = data.features

            n_rows = features.shape[0]
            n_cat = len(self.categorical_idx_)

            filled = xp.full((n_rows, n_cat), 1.0, dtype=float)

            features[:, self.categorical_idx_] = filled
            data.features = features

            return data

    custom_strategy = [{
        "method": 'OneFiller',
        "features_idx": ["B"],
        "implementation": OnesFiller,
        "step_args": None,
    }]

    td = TensorDataCreator.create(X, backend_name="cpu",
                                  features_names=features_names,
                                  custom_strategy=custom_strategy)

    np_features = td.features.numpy()

    # col "C" was encoded using label encoding
    ref_X = np.array([
        [100, 1, 0],
        [500, 1, 1],
        [100, 1, 0],
    ], dtype=np.float32)

    assert isinstance(td, TensorData)
    assert np.allclose(np_features, ref_X, atol=1e-5)
