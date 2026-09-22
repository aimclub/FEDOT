"""Independent properties for evaluation-scoped TensorData identities."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch
from hypothesis import given, strategies as st

from fedot.core.caching.evaluation_context import TensorDataCacheContext, tensor_data_identity
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def tensor_data(rows=257):
    return TensorData(
        Task(TaskTypesEnum.regression),
        DataTypesEnum.table,
        features=torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        target=torch.arange(rows, dtype=torch.float32),
        idx=np.arange(rows),
    )


@given(
    row=st.integers(min_value=0, max_value=256),
    column=st.integers(min_value=0, max_value=1),
    delta=st.integers(min_value=1, max_value=100),
)
def test_identity_observes_every_feature_row(row, column, delta):
    original = tensor_data()
    changed = deepcopy(original)
    changed.features[row, column] += delta

    assert tensor_data_identity(original) != tensor_data_identity(changed)


@given(row=st.integers(min_value=0, max_value=256), delta=st.integers(min_value=1, max_value=100))
def test_fold_context_observes_every_target_value(row, delta):
    train, test = tensor_data(), tensor_data()
    changed = deepcopy(train)
    changed.target[row] += delta

    original_context = TensorDataCacheContext.from_fold(
        train, test, 0, 'candidate')
    changed_context = TensorDataCacheContext.from_fold(
        changed, test, 0, 'candidate')

    assert original_context.data_id != changed_context.data_id
    assert original_context.key != changed_context.key


@st.composite
def context_variations(draw):
    field = draw(st.sampled_from([
        'data_id', 'preparation_id', 'fold_id', 'candidate_id', 'namespace', 'backend']))
    values = {
        'data_id': st.text(alphabet='abc123', min_size=1, max_size=12).filter(lambda value: value != 'data'),
        'preparation_id': st.text(alphabet='abc123', min_size=1, max_size=12).filter(
            lambda value: value != 'preparation'),
        'fold_id': st.integers(min_value=1, max_value=20),
        'candidate_id': st.text(alphabet='abc123', min_size=1, max_size=12).filter(
            lambda value: value != 'candidate'),
        'namespace': st.text(alphabet='abc123-_', min_size=1, max_size=12).filter(
            lambda value: value != 'namespace'),
        'backend': st.sampled_from(['cuda:0/cuda:0', 'mps/mps']),
    }
    return field, draw(values[field])


@given(variation=context_variations())
def test_context_key_is_stable_and_each_scope_dimension_is_independent(variation):
    base = TensorDataCacheContext(
        'data', 'preparation', 0, 'candidate', 'namespace', 'cpu/cpu')
    field, value = variation
    changed = replace(base, **{field: value})

    assert changed.key == deepcopy(changed).key
    assert changed.key != base.key


@given(
    operation_hash=st.text(alphabet='abcdef0123456789',
                           min_size=1, max_size=32),
    first_state=st.sampled_from(['fit', 'predict', 'transform']),
    second_state=st.sampled_from(['fit', 'predict', 'transform']),
)
def test_operation_key_separates_pipeline_state(operation_hash, first_state, second_state):
    context = TensorDataCacheContext('data', 'preparation', 0, 'candidate')

    first = context.operation_key(operation_hash, first_state)
    second = context.operation_key(operation_hash, second_state)

    assert first == context.operation_key(operation_hash, first_state)
    assert (first == second) is (first_state == second_state)
