from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from fedot import create_data
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.evaluation_context import TensorDataCacheContext, tensor_data_identity
from fedot.core.caching.evaluation_session import PredictionCacheSession
from fedot.core.caching.inmemory_operations import save_tensor_data
from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.planner import PreprocessingPlan


def data():
    return TensorData(Task(TaskTypesEnum.regression), DataTypesEnum.table,
                      features=torch.arange(400, dtype=torch.float32).reshape(200, 2),
                      target=torch.arange(200, dtype=torch.float32), idx=np.arange(200))


def context(td):
    return TensorDataCacheContext.from_fold(td, deepcopy(td), 0, 'model')


@pytest.mark.parametrize('field', ['features', 'target', 'idx', 'predict', 'features_names'])
def test_full_content_identity_detects_semantic_change(field):
    original = data()
    changed = deepcopy(original)
    if field in ('features', 'target', 'idx'):
        getattr(changed, field)[99] += 1
    elif field == 'predict':
        changed.predict = torch.ones(200)
    else:
        changed.features_names = ['a', 'b']
    assert tensor_data_identity(original) != tensor_data_identity(changed)
    assert context(original).key != context(changed).key


def test_identity_ignores_derived_trace_fields_and_is_stable_for_copies():
    original = data()
    changed = deepcopy(original)
    changed.fingerprint, changed.trace_uuid = 'obsolete', 'new-trace'
    assert tensor_data_identity(original) == tensor_data_identity(changed)


@pytest.mark.parametrize('field,value', [('candidate_id', 'other'), ('fold_id', 1), ('namespace', 'new-version'),
                                         ('preparation_id', 'new-plan'), ('backend', 'other-device'), ('data_id', 'new-data')])
def test_context_dimensions_never_alias(field, value):
    first = context(data())
    assert first.key != replace(first, **{field: value}).key


def test_context_rejects_boolean_fold_id():
    with pytest.raises(ValueError, match='fold_id'):
        TensorDataCacheContext('data', 'preparation', True, 'candidate')


def test_preparation_snapshot_changes_identity():
    td = create_data(np.array([[1., 2.], [3., 4.]]), target=np.array([0, 1]), use_cache=False)
    changed = deepcopy(td)
    changed.preparation_state = replace(td.preparation_state, input_hash='other-training-input')
    assert tensor_data_identity(td) != tensor_data_identity(changed)


def test_scoped_tensor_cache_roundtrip_and_context_miss(isolated_cache_dir):
    td, operation = data(), PreprocessingPlan()
    scope = context(td)
    cache = Cacher(CacheIndexDB())
    record = cache.cache_tensor_data(td, input_data=td, operation=operation, context=scope)
    loaded = cache.load_tensor_data(td, operation, context=scope)
    assert record.path.exists()
    assert loaded.success
    assert tensor_data_identity(loaded.data) == tensor_data_identity(td)
    assert not cache.load_tensor_data(td, operation, context=replace(scope, fold_id=1)).success
    assert not cache.load_tensor_data(td, operation).success
    assert not cache.load_tensor_data(td, operation, context=scope, state='predict').success


def test_prepared_tensor_cache_roundtrip_preserves_training_state(isolated_cache_dir):
    td = create_data(np.array([[1., 2.], [3., 4.]]), target=np.array([0, 1]), use_cache=False)
    operation, cache = PreprocessingPlan(), Cacher(CacheIndexDB())
    scope = context(td)
    cache.cache_tensor_data(td, input_data=td, operation=operation, context=scope)
    loaded = cache.load_tensor_data(td, operation, context=scope)
    assert loaded.success
    assert loaded.data.preparation_state == td.preparation_state
    assert tensor_data_identity(loaded.data) == tensor_data_identity(td)


def test_scoped_cache_rejects_corrupt_content_even_with_matching_stored_fingerprint(isolated_cache_dir):
    td, operation = data(), PreprocessingPlan()
    scope, cache = context(td), Cacher(CacheIndexDB())
    record = cache.cache_tensor_data(td, input_data=td, operation=operation, context=scope)
    corrupt = deepcopy(td)
    corrupt.features[99, 0] = -1000
    record.path.unlink()
    save_tensor_data(corrupt, record.output_hash)
    with pytest.raises(ValueError, match='complete content identity'):
        cache.load_tensor_data(td, operation, context=scope)


def test_disabled_scoped_cache_has_no_file_and_reports_miss(isolated_cache_dir):
    td, operation = data(), PreprocessingPlan()
    scope, cache = context(td), Cacher(CacheIndexDB(), use_cache=False)
    assert cache.cache_tensor_data(td, input_data=td, operation=operation, context=scope).path is None
    assert not cache.load_tensor_data(td, operation, context=scope).success


def test_context_only_lookup_uses_declared_dataset_identity(isolated_cache_dir):
    td, operation = data(), PreprocessingPlan()
    scope, cache = context(td), Cacher(CacheIndexDB())
    cache.cache_tensor_data(td, operation=operation, context=scope)
    assert cache.load_tensor_data(None, operation, context=scope).success


def test_raw_scoped_cache_includes_target_values(isolated_cache_dir):
    td, operation = data(), PreprocessingPlan()
    scope, cache = context(td), Cacher(CacheIndexDB())
    features, target = np.ones((2, 2)), np.array([0, 1])
    cache.cache_tensor_data(td, input_data=features, target=target, operation=operation, context=scope)
    assert cache.load_tensor_data(features, operation, target, context=scope).success
    assert not cache.load_tensor_data(features, operation, target[::-1], context=scope).success


def test_prediction_session_discards_failed_attempt_and_isolates_values():
    from unittest.mock import Mock
    backend = Mock()
    session = PredictionCacheSession(backend)
    td = data()
    session.save_node_prediction('node', 'raw', 'fold', td)
    loaded = session.load_node_prediction('node', 'raw', 'fold')
    loaded.features[0, 0] = -1
    assert session.load_node_prediction('node', 'raw', 'fold').features[0, 0] == 0
    backend.save_node_prediction.assert_not_called()
    session.close()
    assert session.pending == {} and session.cache is None


def test_prediction_session_commits_once_then_releases_references():
    from unittest.mock import Mock
    backend = Mock()
    session = PredictionCacheSession(backend)
    session.save_node_prediction('node', 'raw', 'fold', data())
    session.commit()
    assert backend.save_node_prediction.call_count == 1
    session.close()
    assert session.pending == {}
