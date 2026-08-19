import numpy as np
import pytest
import torch

from fedot import Fedot, TensorData, create_data
from fedot.api.create_data import _build_create_data_kwargs, _resolve_target_argument
from fedot.core.caching.cacher import Cacher
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.validation.errors import FedotValidationError


@pytest.mark.unit
def test_resolve_target_argument_maps_column_name_to_target_idx():
    target, target_idx = _resolve_target_argument('label', None)
    assert target is None
    assert target_idx == 'label'


@pytest.mark.unit
def test_resolve_target_argument_rejects_column_name_with_target_idx():
    with pytest.raises(ValueError, match='not both'):
        _resolve_target_argument('label', 0)


@pytest.mark.unit
def test_build_create_data_kwargs_from_data_sets_predict_defaults():
    train = TensorData(
        task='classification',
        data_type='tabular',
        features=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        target=torch.tensor([0.0]),
        trace_uuid='trace-abc',
    )

    backend_name, spec_kwargs = _build_create_data_kwargs(from_data=train)

    assert backend_name == 'cpu'
    assert spec_kwargs['state'] is StateEnum.PREDICT
    assert spec_kwargs['task'] is train.task
    assert spec_kwargs['data_type'] is train.data_type
    assert spec_kwargs['trace_uuid'] == 'trace-abc'


@pytest.mark.unit
def test_create_data_fit_and_from_data_predict(isolated_cache_dir):
    features = np.array([
        [1.0, 0.1, 'cat'],
        [2.0, 0.2, 'dog'],
        [3.0, 0.3, 'cat'],
        [4.0, 0.4, 'dog'],
    ], dtype=object)

    train = create_data(features)
    assert isinstance(train, TensorData)
    assert train.state is StateEnum.FIT
    assert train.trace_uuid is not None
    assert train.features.shape[0] == 4
    assert train.target is not None
    assert train.features.shape[1] < features.shape[1]

    test_features = np.array([
        [5.0, 0.5],
        [6.0, 0.6],
    ], dtype=object)
    test = create_data(test_features, from_data=train)

    assert test.state is StateEnum.PREDICT
    assert test.trace_uuid == train.trace_uuid
    assert test.task.task_type is train.task.task_type
    assert test.target is None


@pytest.mark.unit
def test_create_data_accepts_explicit_target_array(isolated_cache_dir):
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)

    train = create_data(x, target=y)

    assert train.features.shape == (2, 2)
    assert train.target is not None
    assert train.target.shape[0] == 2


@pytest.mark.unit
def test_fedot_create_data_uses_problem_and_tensor_data_config(isolated_cache_dir):
    model = Fedot(
        problem='classification',
        use_cache=False,
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    train = model.create_data(x, target=y)

    assert train.task.task_type is TaskTypesEnum.classification
    assert train.state is StateEnum.FIT

    test = model.create_data(x[:2], from_data=train)
    assert test.state is StateEnum.PREDICT
    assert test.trace_uuid == train.trace_uuid
    assert Cacher().use_cache is False


@pytest.mark.unit
def test_tensor_data_creator_use_cache_sets_runtime_cacher(isolated_cache_dir):
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    Cacher().set(use_cache=True)
    TensorDataCreator.create(x, backend_name='cpu', target=y, use_cache=False)
    assert Cacher().use_cache is False

    create_data(x, target=y, use_cache=True)
    assert Cacher().use_cache is True


@pytest.mark.unit
def test_fedot_tensor_data_config_use_cache_overrides_api_flag(isolated_cache_dir):
    model = Fedot(
        problem='classification',
        use_cache=True,
        tensor_data_config={'use_cache': False},
    )
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    model.create_data(x, target=y)
    assert Cacher().use_cache is False


@pytest.mark.unit
def test_fedot_create_data_predict_requires_trace_uuid():
    model = Fedot(problem='classification')
    train_without_trace = TensorData(
        task='classification',
        data_type='tabular',
        features=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        trace_uuid=None,
    )

    with pytest.raises(FedotValidationError, match='trace_uuid is required'):
        model.create_data(
            np.array([[1.0, 2.0]], dtype=np.float32),
            from_data=train_without_trace,
        )


@pytest.mark.unit
def test_public_exports():
    import fedot
    from fedot.api import create_data as api_create_data

    assert fedot.create_data is create_data
    assert api_create_data is create_data
    assert fedot.TensorData is TensorData
