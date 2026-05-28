import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from fedot.core.caching.inmemory_operations import save_preprocessing_model
from fedot.core.caching.normalization import prepare_value_for_torch_save
from fedot.core.caching.rules import SaverNotFoundError
from fedot.core.caching.cache_saver import Saver
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class PickleableCustomPreprocessor(AbstractPreprocessingHandler):
    def __init__(self, fitted_tensor: torch.Tensor):
        self.fitted_tensor = fitted_tensor
        self.numpy_state = np.array([1.0, 2.0])
        self.nested_state = {"tensor": fitted_tensor, "label": "state"}

    def fit(self, data, features_idx):
        return self

    def transform(self, data):
        return data


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_tensor_data() -> TensorData:
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_()
    target = torch.tensor([0, 1, 0], dtype=torch.long)

    tensor_data = TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=features,
        target=target,
        categorical_idx=[1],
        numerical_idx=[0],
    )
    tensor_data.predict = {
        "scores": torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32),
    }
    tensor_data.features_names = np.array(["a", "b"])
    tensor_data.custom_strategy = {
        "path": Path("data/source.csv"),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    return tensor_data


@pytest.mark.unit
def test_saver_saves_tensor_data_as_normalized_torch_payload():
    tensor_data = _make_tensor_data()

    response = Saver.save(tensor_data, "tensor-key")

    payload = _torch_load(response.path)
    fields = payload["fields"]

    assert response.success is True
    assert response.kind == "tensor_data"
    assert response.path.exists()
    assert payload["format"] == "fedot-tensor-data-cache-v1"
    assert payload["class_path"].endswith(".TensorData")
    assert fields["features"].device.type == "cpu"
    assert fields["features"].requires_grad is False
    assert torch.equal(fields["features"], tensor_data.features.detach())
    assert torch.equal(fields["target"], tensor_data.target)
    assert torch.equal(fields["predict"]["scores"], tensor_data.predict["scores"])
    assert fields["task"]["class_path"].endswith(".Task")
    assert fields["task"]["fields"]["task_type"] == TaskTypesEnum.classification.value
    assert fields["data_type"] == DataTypesEnum.table.value
    assert np.array_equal(fields["features_names"], tensor_data.features_names)
    assert fields["custom_strategy"]["path"] == "data/source.csv"
    assert fields["custom_strategy"]["dtype"] == "torch.float32"
    assert fields["custom_strategy"]["device"] == "cpu"


@pytest.mark.unit
def test_prepare_value_for_torch_save_avoids_unneeded_array_and_cpu_tensor_copies():
    array = np.array([1, 2, 3])
    tensor = torch.tensor([1.0, 2.0, 3.0])

    prepared_array = prepare_value_for_torch_save(array)
    prepared_tensor = prepare_value_for_torch_save(tensor)

    assert prepared_array is array
    assert prepared_tensor.device.type == "cpu"
    assert prepared_tensor.data_ptr() == tensor.data_ptr()


@pytest.mark.unit
def test_prepare_value_for_torch_save_moves_cuda_tensor_to_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")

    prepared = prepare_value_for_torch_save(tensor)

    assert prepared.device.type == "cpu"
    assert torch.equal(prepared, tensor.cpu())


@pytest.mark.unit
def test_saver_raises_for_unknown_data_type(monkeypatch):
    monkeypatch.setattr(Saver, "_creators", [(lambda _: False, object())])

    with pytest.raises(SaverNotFoundError, match="No saver function registered"):
        Saver.save(object(), "unknown-key")


@pytest.mark.unit
def test_saver_returns_failed_response_for_unsupported_tensor_data_field():
    tensor_data = _make_tensor_data()
    tensor_data.custom_strategy = {"unsupported": object()}

    response = Saver.save(tensor_data, "bad-tensor-key")

    assert response.success is False
    assert response.kind == "tensor_data"
    assert not response.path.exists()


@pytest.mark.unit
def test_save_preprocessing_model_saves_custom_instance_with_cpu_tensor_state():
    source_tensor = torch.tensor([1.0, 2.0], device="cuda" if torch.cuda.is_available() else "cpu")
    model = PickleableCustomPreprocessor(source_tensor)

    response = save_preprocessing_model(model, "preprocessor-key")

    with open(response.path, "rb") as file:
        restored = pickle.load(file)

    assert response.success is True
    assert response.kind == "preprocessing_model"
    assert isinstance(restored, PickleableCustomPreprocessor)
    assert restored.fitted_tensor.device.type == "cpu"
    assert torch.equal(restored.fitted_tensor, source_tensor.cpu())
    assert restored.nested_state["tensor"].device.type == "cpu"
    assert torch.equal(restored.nested_state["tensor"], source_tensor.cpu())
    assert restored.nested_state["label"] == "state"
    assert restored.numpy_state is not model.numpy_state
    assert np.array_equal(restored.numpy_state, model.numpy_state)
    assert model.fitted_tensor.device == source_tensor.device


@pytest.mark.unit
def test_save_preprocessing_model_returns_failed_response_for_unpickleable_custom_class():

    class LocalCustomPreprocessor(AbstractPreprocessingHandler):
        def fit(self, data, features_idx):
            return self

        def transform(self, data):
            return data

    response = save_preprocessing_model(LocalCustomPreprocessor(), "bad-preprocessor-key")

    assert response.success is False
    assert response.kind == "preprocessing_model"
    assert not response.path.exists()
