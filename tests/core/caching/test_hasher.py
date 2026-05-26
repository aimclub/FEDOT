import numpy as np
import pytest
import torch

from fedot.core.caching import Hasher
from fedot.core.caching.rules import HasherNotFoundError
from fedot.core.caching.tools import tensor_features_for_hash
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
def test_raw_array_hash_is_stable_for_equal_numpy_arrays():
    features = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    same_features = features.copy()

    assert Hasher.hash(features) == Hasher.hash(same_features)


@pytest.mark.unit
def test_raw_array_hash_changes_for_different_values_and_dtype():
    features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    changed_values = features.copy()
    changed_values[1, 0] = 30
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
