from pathlib import Path

import numpy as np
import pytest
import torch

from fedot.core.caching import Hasher
from fedot.core.caching.rules import HasherNotFoundError
from fedot.core.caching.tools import (
    build_tensordata_metadata,
    deterministic_positions,
    normalize_for_hash,
    stable_hash,
    tensor_features_for_hash,
)
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.methods.scaling_normalization import MinMaxNormalization, StandartScaling
from fedot.preprocessing.planner import PreprocessingPlan
from fedot.preprocessing.tools.preprocessor_types import (
    PreprocessingStep,
    PreprocessingStepEnum,
    ScalingMethodEnum,
)


def _make_tensor_data(features: torch.Tensor, *, categorical_idx=None) -> TensorData:
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=features,
        categorical_idx=categorical_idx or [],
    )


def _make_scaling_step(
    implementation=None,
    *,
    features_idx=None,
    method=ScalingMethodEnum.standard,
) -> PreprocessingStep:
    return PreprocessingStep(
        step=PreprocessingStepEnum.scaling,
        method=method,
        features_idx=features_idx or [0, 1],
        implementation=implementation,
    )


@pytest.mark.unit
def test_hasher_resolves_first_matching_registered_creator(monkeypatch):
    """Check that Hasher scans registered predicates in order and returns the first matching creator."""
    creator_a = object()
    creator_b = object()

    monkeypatch.setattr(
        Hasher,
        '_creators',
        [
            (lambda _: False, creator_a),
            (lambda _: True, creator_b),
        ],
    )

    assert Hasher.resolve_creator({'source': 'x'}) is creator_b


@pytest.mark.unit
def test_hasher_raises_when_registered_creators_do_not_match(monkeypatch):
    """Check that unknown input types fail with the domain-specific creator lookup error."""
    monkeypatch.setattr(
        Hasher,
        '_creators',
        [(lambda _: False, object())],
    )

    with pytest.raises(HasherNotFoundError, match='No hashing function registered'):
        Hasher.resolve_creator({'source': 'x'})


@pytest.mark.unit
def test_stable_hash_rejects_unsupported_objects():
    class UnsupportedState:
        pass

    with pytest.raises(TypeError, match='Unsupported type for cache hashing'):
        stable_hash({'state': UnsupportedState()})


@pytest.mark.unit
def test_normalize_for_hash_rejects_cyclic_containers():
    cyclic_list = []
    cyclic_list.append(cyclic_list)

    cyclic_dict = {}
    cyclic_dict['self'] = cyclic_dict

    tuple_ref = []
    cyclic_tuple = (tuple_ref,)
    tuple_ref.append(cyclic_tuple)

    for obj in (cyclic_list, cyclic_dict, cyclic_tuple):
        with pytest.raises(TypeError, match='Cycle detected'):
            normalize_for_hash(obj)


@pytest.mark.unit
def test_stable_hash_is_deterministic_and_dict_order_independent():
    first = {'b': [2, 3], 'a': {'nested': True}}
    same_with_different_order = {'a': {'nested': True}, 'b': [2, 3]}
    changed = {'a': {'nested': False}, 'b': [2, 3]}

    assert stable_hash(first) == stable_hash(same_with_different_order)
    assert stable_hash(first) != stable_hash(changed)


@pytest.mark.unit
def test_normalize_for_hash_converts_supported_runtime_values():
    features = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = torch.tensor([[1.0, 2.0]])

    normalized = normalize_for_hash({
        'path': Path('data/file.csv'),
        'method': ScalingMethodEnum.standard,
        'features': features,
        'tensor': tensor,
    })

    assert normalized['path'] == 'data/file.csv'
    assert normalized['method'] == ScalingMethodEnum.standard.value
    assert normalized['features']['shape'] == (2, 2)
    assert normalized['features']['dtype'] == 'int64'
    assert normalized['tensor']['shape'] == (1, 2)
    assert normalized['tensor']['dtype'] == 'torch.float32'


@pytest.mark.unit
def test_deterministic_positions_are_stable_sorted_and_include_anchors():
    seed_data = {'shape': (128, 8), 'dtype': 'float32'}

    positions = deterministic_positions(
        total_rows=128,
        n_samples=16,
        seed_data=seed_data,
    )
    same_positions = deterministic_positions(
        total_rows=128,
        n_samples=16,
        seed_data=seed_data,
    )

    assert positions == same_positions
    assert positions == sorted(positions)
    assert len(positions) == 16
    assert {0, 64, 127}.issubset(positions)


@pytest.mark.unit
def test_build_tensordata_metadata_includes_shape_dtype_and_indices():
    features = torch.zeros((3, 2), dtype=torch.float64)
    target = torch.ones(3, dtype=torch.float32)
    td = _make_tensor_data(features, categorical_idx=[1])
    td.target = target
    td.numerical_idx = [0]
    td.features_names = np.array(['a', 'b'])
    td.idx_mapping = {0: 10, 1: 11, 2: 12}

    metadata = build_tensordata_metadata(td)

    assert metadata['features_shape'] == (3, 2)
    assert metadata['features_dtype'] == 'torch.float64'
    assert metadata['target_shape'] == (3,)
    assert metadata['target_dtype'] == 'torch.float32'
    assert metadata['categorical_idx'] == [1]
    assert metadata['numerical_idx'] == [0]
    assert np.asarray(metadata['features_names']).tolist() == ['a', 'b']
    assert metadata['idx_mapping'] == {0: 10, 1: 11, 2: 12}


@pytest.mark.unit
def test_raw_array_hash_is_stable_for_equal_numpy_arrays():
    features = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    same_features = features.copy()

    assert Hasher.hash(features) == Hasher.hash(same_features)


@pytest.mark.unit
def test_raw_array_hash_changes_for_different_values_and_dtype():
    features = np.arange(256 * 16, dtype=np.int64).reshape(256, 16)
    changed_values = features.copy()
    changed_values[128, 0] = -1
    changed_dtype = features.astype(np.float64)

    assert Hasher.hash(features) != Hasher.hash(changed_values)
    assert Hasher.hash(features) != Hasher.hash(changed_dtype)


@pytest.mark.unit
def test_tensordata_hash_is_stable_for_equal_tensors():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    td = _make_tensor_data(features)
    same_td = _make_tensor_data(features.clone())

    assert Hasher.hash(td) == Hasher.hash(same_td)


@pytest.mark.unit
def test_tensordata_hash_changes_for_feature_values_and_metadata():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    td = _make_tensor_data(features)

    changed_features = _make_tensor_data(features.clone())
    changed_features.features[0, 0] = 10.0

    changed_metadata = _make_tensor_data(features.clone(), categorical_idx=[0])

    assert Hasher.hash(td) != Hasher.hash(changed_features)
    assert Hasher.hash(td) != Hasher.hash(changed_metadata)


@pytest.mark.unit
def test_tensordata_hash_changes_for_idx_mapping_metadata():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    td = _make_tensor_data(features)
    same_features_changed_mapping = _make_tensor_data(features.clone())
    same_features_changed_mapping.idx_mapping = {0: 10, 1: 11}

    assert Hasher.hash(td) != Hasher.hash(same_features_changed_mapping)


@pytest.mark.unit
def test_tensordata_hash_samples_channel_tensors_as_rows():
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    td = _make_tensor_data(features)

    changed_channel_value = _make_tensor_data(features.clone())
    changed_channel_value.features[1, 2, 3] = -1.0

    assert tuple(tensor_features_for_hash(features).shape) == (6, 4)
    assert Hasher.hash(td) != Hasher.hash(changed_channel_value)


@pytest.mark.unit
def test_preprocessing_handler_hash_uses_class_and_instance_state():
    scaling = StandartScaling()
    same_scaling = StandartScaling()
    changed_scaling = StandartScaling(with_mean=False)

    assert Hasher.hash(StandartScaling) != Hasher.hash(MinMaxNormalization)
    assert Hasher.hash(scaling) == Hasher.hash(same_scaling)
    assert Hasher.hash(scaling) != Hasher.hash(changed_scaling)


@pytest.mark.unit
def test_preprocessing_handler_hash_changes_for_fitted_attributes():
    scaling = StandartScaling()
    same_scaling = StandartScaling()
    changed_scaling = StandartScaling()

    scaling.mean_values = torch.tensor([1.0, 2.0])
    same_scaling.mean_values = torch.tensor([1.0, 2.0])
    changed_scaling.mean_values = torch.tensor([2.0, 3.0])

    assert Hasher.hash(scaling) == Hasher.hash(same_scaling)
    assert Hasher.hash(scaling) != Hasher.hash(changed_scaling)


@pytest.mark.unit
def test_preprocessing_plan_hash_is_order_sensitive():
    first_step = _make_scaling_step(StandartScaling(), features_idx=[0])
    second_step = _make_scaling_step(MinMaxNormalization(), features_idx=[1], method=ScalingMethodEnum.min_max)

    plan = PreprocessingPlan([first_step, second_step])
    same_plan = PreprocessingPlan([
        _make_scaling_step(StandartScaling(), features_idx=[0]),
        _make_scaling_step(MinMaxNormalization(), features_idx=[1], method=ScalingMethodEnum.min_max),
    ])
    reordered_plan = PreprocessingPlan([second_step, first_step])

    assert Hasher.hash(plan) == Hasher.hash(same_plan)
    assert Hasher.hash(plan) != Hasher.hash(reordered_plan)


@pytest.mark.unit
def test_preprocessing_plan_hash_changes_for_step_configuration_and_implementation_state():
    scaling = StandartScaling()
    changed_scaling = StandartScaling()
    changed_scaling.scale_values = torch.tensor([10.0])

    plan = PreprocessingPlan([_make_scaling_step(scaling, features_idx=[0])])
    changed_features_idx_plan = PreprocessingPlan([_make_scaling_step(StandartScaling(), features_idx=[1])])
    changed_state_plan = PreprocessingPlan([_make_scaling_step(changed_scaling, features_idx=[0])])

    assert Hasher.hash(plan) != Hasher.hash(changed_features_idx_plan)
    assert Hasher.hash(plan) != Hasher.hash(changed_state_plan)


@pytest.mark.unit
def test_hasher_raises_for_unsupported_objects():
    with pytest.raises(HasherNotFoundError, match="No hashing function registered"):
        Hasher.hash(object())
