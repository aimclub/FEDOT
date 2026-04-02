import numpy as np
import pytest

from fedot.core.backend.backend import backend
from fedot.core.data.tensordata import LoadDataSpec, TensorData, from_csv_tsv, from_numpy
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.tensordata_rules import TensorDataCreatorNotFoundError


@pytest.mark.unit
def test_resolve_creator_uses_registered_predicates_in_order(monkeypatch):
    creator_a = object()
    creator_b = object()

    monkeypatch.setattr(
        TensorData,
        '_creators',
        [
            (lambda _: False, creator_a),
            (lambda _: True, creator_b),
        ],
    )

    assert TensorData._resolve_creator({'source': 'x'}) is creator_b


@pytest.mark.unit
def test_resolve_creator_raises_when_no_creator_matches(monkeypatch):
    monkeypatch.setattr(
        TensorData,
        '_creators',
        [(lambda _: False, object())],
    )

    with pytest.raises(TensorDataCreatorNotFoundError, match='No creator registered'):
        TensorData._resolve_creator({'source': 'x'})


@pytest.mark.unit
def test_create_normalizes_backend_and_builds_typed_spec(monkeypatch):
    backend_calls = []

    def fake_backend_set(name):
        backend_calls.append(name)

    def fake_creator(_source_data, spec):
        return spec

    monkeypatch.setattr(backend, 'set', fake_backend_set)
    monkeypatch.setattr(TensorData, '_resolve_creator', lambda _source: fake_creator)

    spec = TensorData.create({'source': 'x'}, backend_name=' GPU ', task='classification', state='predict')

    assert backend_calls == ['gpu']
    assert isinstance(spec, LoadDataSpec)
    assert spec.task.task_type.value == 'classification'
    assert spec.state.value == 'predict'


@pytest.mark.unit
def test_create_wraps_creation_error_with_source_and_backend_context(monkeypatch):
    def fake_backend_set(_name):
        return None

    def raising_creator(_source_data, _spec):
        raise RuntimeError('boom')

    monkeypatch.setattr(backend, 'set', fake_backend_set)
    monkeypatch.setattr(TensorData, '_resolve_creator', lambda _source: raising_creator)

    with pytest.raises(ValueError, match='source_type=dict'):
        TensorData.create({'source': 'x'}, backend_name='cpu')


@pytest.mark.unit
def test_from_numpy_treats_integer_target_as_target_idx_without_copy_bug():
    spec = LoadDataSpec(target=1)

    captured = {}

    def fake_to_tensor_data(features):
        captured['features_shape'] = features.shape
        captured['target'] = spec.target
        captured['target_idx'] = spec.target_idx
        return 'created'

    spec.to_tensor_data = fake_to_tensor_data

    result = from_numpy(features=np.array([[1, 2, 3], [4, 5, 6]]), spec=spec)

    assert result == 'created'
    assert captured['features_shape'] == (2, 3)
    assert captured['target'] is None
    assert captured['target_idx'] == 1


@pytest.mark.unit
def test_from_csv_tsv_uses_file_load_plan_defaults(monkeypatch, tmp_path):
    csv_path = tmp_path / 'sample.tsv'
    csv_path.write_text('idx\tvalue\n1\t2\n')

    captured = {}

    def fake_get_df_from_csv(file_path, delimiter, index_col, possible_idx_keywords, columns_to_drop, nrows):
        captured['file_path'] = file_path
        captured['delimiter'] = delimiter
        captured['possible_idx_keywords'] = possible_idx_keywords

        class _Frame:
            columns = np.array(['value'])
            values = np.array([[2]])

        return _Frame()

    monkeypatch.setattr('fedot.core.data.tensordata.get_df_from_csv', fake_get_df_from_csv)
    monkeypatch.setattr('fedot.core.data.tensordata.get_values_from_df', lambda frame: frame.values)

    spec = LoadDataSpec()
    spec.to_tensor_data = lambda features: features

    result = from_csv_tsv(str(csv_path), spec)

    assert result.shape == (1, 1)
    assert captured['file_path'] == str(csv_path)
    assert captured['delimiter'] == '\t'
    assert captured['possible_idx_keywords'] == ['idx', 'index', 'id', 'unnamed: 0']


@pytest.mark.unit
def test_load_data_spec_applies_typed_normalization_and_default_merges():
    spec = LoadDataSpec(
        task='classification',
        data_type='image',
        state='predict',
        ts_orientation='long',
        embedding_strategy=None,
        dataloader_kwargs={'batch_size': 16},
    )

    assert spec.task.task_type.value == 'classification'
    assert spec.data_type.value == 'time_series'
    assert spec.state.value == 'predict'
    assert spec.ts_orientation.value == 'long'
    assert spec.embedding_strategy == {}
    assert spec.dataloader_kwargs == {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0,
        'drop_last': False,
    }


@pytest.mark.unit
def test_load_data_spec_normalization_is_deterministic_on_recreation():
    first = LoadDataSpec(
        task='classification',
        data_type='table',
        state='fit',
        ts_orientation=None,
        embedding_strategy={'method': 'demo'},
        dataloader_kwargs={'shuffle': False},
    )

    second = LoadDataSpec(
        task=first.task,
        data_type=first.data_type,
        state=first.state,
        ts_orientation=first.ts_orientation,
        embedding_strategy=first.embedding_strategy,
        dataloader_kwargs=first.dataloader_kwargs,
    )

    assert second.task.task_type == first.task.task_type
    assert second.data_type == first.data_type
    assert second.state == first.state
    assert second.ts_orientation == first.ts_orientation
    assert second.embedding_strategy == first.embedding_strategy
    assert second.dataloader_kwargs == first.dataloader_kwargs
    assert second.embedding_strategy is not first.embedding_strategy
    assert second.dataloader_kwargs is not first.dataloader_kwargs


@pytest.mark.unit
def test_from_numpy_autodetects_tensor_canonical_tabular_when_data_type_missing():
    spec = LoadDataSpec(data_type=None, task='classification')

    captured = {}

    def fake_to_tensor_data(features):
        captured['data_type'] = spec.data_type
        return features

    spec.to_tensor_data = fake_to_tensor_data

    from_numpy(features=np.array([[1, 2], [3, 4]]), spec=spec)

    assert captured['data_type'].value == 'table'
    assert captured['data_type'] == captured['data_type'].tabular


@pytest.mark.unit
def test_from_numpy_autodetects_tensor_canonical_ts_for_forecasting_when_data_type_missing():
    spec = LoadDataSpec(data_type=None, task=Task(TaskTypesEnum.ts_forecasting))

    captured = {}

    def fake_to_tensor_data(features):
        captured['data_type'] = spec.data_type
        return features

    spec.to_tensor_data = fake_to_tensor_data

    from_numpy(features=np.array([[1, 2], [3, 4]]), spec=spec)

    assert captured['data_type'].value == 'time_series'
    assert captured['data_type'] == captured['data_type'].ts


@pytest.mark.unit
def test_post_init_raw_builds_default_idx_from_sample_axis(monkeypatch):
    tensor_data = TensorData.__new__(TensorData)
    tensor_data.task = Task(TaskTypesEnum.classification)
    tensor_data.data_type = None
    tensor_data.state = 'fit'
    tensor_data.idx = None
    tensor_data.features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor_data.target = np.array([0, 1])
    tensor_data.predict = None
    tensor_data.target_idx = None
    tensor_data.target_encoder = None
    tensor_data.categorical_idx = None
    tensor_data.encoding_strategy = None
    tensor_data.text_idx = None
    tensor_data.embedding_strategy = {}
    tensor_data.features_names = None
    tensor_data.ts_orientation = None
    tensor_data.ts_terms_idx = None
    tensor_data.ts_forecast_horizon = None
    tensor_data.ts_init_shape = None
    tensor_data.dataloader_kwargs = {}

    monkeypatch.setattr('fedot.core.data.tensordata.replace_missing_with_nan', lambda features: features)
    monkeypatch.setattr(
        'fedot.core.data.tensordata.process_ts_data',
        lambda features, target, features_names, state, ts_orientation, ts_terms_idx, ts_forecast_horizon, data_type: (features, target, None, ts_terms_idx),
    )
    monkeypatch.setattr(
        'fedot.core.data.tensordata.get_target_and_features',
        lambda features, target, features_names, target_idx, state, data_type: (features, target, None),
    )
    monkeypatch.setattr(
        'fedot.core.data.tensordata.get_text_embeddings',
        lambda features, text_idx, embedding_strategy, features_names: (None, text_idx, features),
    )
    monkeypatch.setattr(
        'fedot.core.data.tensordata.choose_categorical_encoding',
        lambda features, categorical_idx, encoding_strategy, features_names, state: ({}, None),
    )
    monkeypatch.setattr(
        'fedot.core.data.tensordata.encode_categorical_features',
        lambda features, encoding_decisions, non_cat_features: (features, None),
    )
    monkeypatch.setattr(
        'fedot.core.data.tensordata.transform_to_tensor',
        lambda features, target, text_tensors, text_idx, ts_init_shape: (features, target),
    )

    tensor_data._post_init_raw()

    assert np.array_equal(tensor_data.idx, np.array([0, 1]))

