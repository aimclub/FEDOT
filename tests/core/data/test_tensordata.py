import numpy as np
import pytest

from fedot.core.backend.backend import backend
from fedot.core.data.tensordata import LoadDataSpec, TensorData, from_csv_tsv, from_numpy
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
