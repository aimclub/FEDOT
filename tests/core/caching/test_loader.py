import uuid

import numpy as np
import pytest
import torch

from fedot.core.backend.backend import Backend
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.cache_saver import Saver
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.inmemory_operations import load_pkl_file
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class PickleableLoaderPreprocessor(AbstractPreprocessingHandler):
    def __init__(self, fitted_tensor: torch.Tensor):
        self.fitted_tensor = fitted_tensor
        self.nested_state = {"tensor": fitted_tensor}

    def fit(self, data, features_idx):
        return self

    def transform(self, data):
        return data


def _make_key(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _make_tensor_data() -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        target=torch.tensor([0, 1, 0], dtype=torch.long),
        categorical_idx=[1],
        numerical_idx=[0],
        features_names=["a", "b"],
    )


@pytest.mark.unit
def test_loader_restores_tensor_data_and_validates_hash():
    tensor_data = _make_tensor_data()
    expected_hash = Hasher.hash(tensor_data)
    response = Saver.save(tensor_data, _make_key("tensor"))

    loaded = Loader.load(str(response.path), hash=expected_hash)

    assert isinstance(loaded, TensorData)
    assert loaded.features.device == Backend().device
    assert loaded == tensor_data
    assert Hasher.hash(loaded) == expected_hash


@pytest.mark.unit
def test_loader_rejects_tensor_data_hash_mismatch():
    response = Saver.save(_make_tensor_data(), _make_key("tensor"))

    with pytest.raises(ValueError, match="Loaded cache hash mismatch"):
        Loader.load(str(response.path), hash="wrong-hash")


@pytest.mark.unit
def test_load_pkl_file_restores_preprocessing_model_to_backend_and_validates_hash():
    source_tensor = torch.tensor([1.0, 2.0])
    model = PickleableLoaderPreprocessor(source_tensor)
    expected_hash = Hasher.hash(model)
    response = Saver.save(model, _make_key("model"))

    loaded = load_pkl_file(str(response.path), hash=expected_hash, kind=None)

    assert isinstance(loaded, PickleableLoaderPreprocessor)
    assert loaded.fitted_tensor.device == Backend().device
    assert loaded.nested_state["tensor"].device == Backend().device
    assert torch.equal(loaded.fitted_tensor.cpu(), source_tensor)
    assert Hasher.hash(loaded) == expected_hash
