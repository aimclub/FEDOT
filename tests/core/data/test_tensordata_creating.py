import numpy as np
import pytest

from fedot.core.backend.backend import Backend
from fedot.core.data.reader.data_reader import DataReader, from_csv_tsv, from_numpy
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.data.tensor_data.rules import TensorDataCreatorNotFoundError
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_data_reader_resolves_first_matching_registered_reader(monkeypatch):
    """Check that DataReader scans registered predicates in order and returns the first matching reader."""
    creator_a = object()
    creator_b = object()

    monkeypatch.setattr(
        DataReader,
        '_creators',
        [
            (lambda _: False, creator_a),
            (lambda _: True, creator_b),
        ],
    )

    assert DataReader._resolve_creator({'source': 'x'}) is creator_b


@pytest.mark.unit
def test_data_reader_raises_when_registered_readers_do_not_match(monkeypatch):
    """Check that unknown input types fail with the domain-specific creator lookup error."""
    monkeypatch.setattr(
        DataReader,
        '_creators',
        [(lambda _: False, object())],
    )

    with pytest.raises(TensorDataCreatorNotFoundError, match='No creator registered'):
        DataReader._resolve_creator({'source': 'x'})


@pytest.mark.unit
def test_tensor_data_creator_normalizes_backend_and_initializes_data_spec(monkeypatch):
    """Check the outer creation flow without running expensive preprocessing.

    The creator should normalize backend names, initialize DataSpec from kwargs, and pass
    the spec through reader/preprocessing/conversion stages.
    """
    backend_calls = []

    def fake_backend_set(self, name):
        backend_calls.append(name)

    def fake_read(self, source_data, spec):
        spec.features = source_data['features']
        return spec

    monkeypatch.setattr(Backend, 'set', fake_backend_set)
    monkeypatch.setattr(DataReader, 'read', fake_read)
    monkeypatch.setattr(TensorDataCreator, 'preprocess_data', lambda self: None)
    monkeypatch.setattr(TensorDataCreator, 'to_tensor_data', lambda self: self.spec)
    monkeypatch.setattr(TensorDataCreator, 'to_backend', lambda self, tensor_data: tensor_data)

    spec = TensorDataCreator.create(
        {'features': np.array([[1, 2], [3, 4]])},
        backend_name=' GPU ',
        task='classification',
        state='predict',
    )

    assert backend_calls == ['gpu']
    assert isinstance(spec, DataSpec)
    assert spec.task.task_type.value == 'classification'
    assert spec.state.value == 'predict'


@pytest.mark.unit
def test_tensor_data_creator_wraps_reader_errors_with_creation_context(monkeypatch):
    """Check that low-level reader failures are reported with source type and backend context."""
    def fake_backend_set(self, _name):
        return None

    def raising_read(self, _source_data, _spec):
        raise RuntimeError('boom')

    monkeypatch.setattr(Backend, 'set', fake_backend_set)
    monkeypatch.setattr(DataReader, 'read', raising_read)

    with pytest.raises(ValueError, match='source_type=dict'):
        TensorDataCreator.create({'source': 'x'}, backend_name='cpu')


@pytest.mark.unit
def test_numpy_reader_only_stores_raw_features_in_data_spec():
    """Check that the numpy reader does not perform target normalization or preprocessing.

    Target-to-target_idx conversion belongs to TensorDataCreator.preprocess_data, not to
    DataReader.from_numpy.
    """
    spec = DataSpec(target=1)
    features = np.array([[1, 2, 3], [4, 5, 6]])

    result = from_numpy(features=features, spec=spec)

    assert result is spec
    assert result.features is features
    assert result.target == 1
    assert result.target_idx is None


@pytest.mark.unit
def test_preprocess_data_converts_integer_target_reference_to_target_idx(monkeypatch):
    """Check that integer `target` values are interpreted as target column references.

    The feature matrix must stay untouched at this stage, while `target` is cleared and
    `target_idx` receives the column index.
    """
    Backend().set('cpu')
    creator = TensorDataCreator()
    creator.spec = DataSpec(
        features=np.array([[1, 2, 3], [4, 5, 6]]),
        target=1,
    )

    monkeypatch.setattr(TensorDataCreator, 'obligatory_preprocess', lambda self: None)

    creator.preprocess_data()

    assert creator.spec.features.shape == (2, 3)
    assert creator.spec.target is None
    assert creator.spec.target_idx == 1


@pytest.mark.unit
def test_csv_tsv_reader_applies_file_load_plan_defaults(monkeypatch, tmp_path):
    """Check CSV/TSV reader planning: extension-based delimiter and default index keywords."""
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

    monkeypatch.setattr('fedot.core.data.reader.data_reader.get_df_from_csv', fake_get_df_from_csv)
    monkeypatch.setattr('fedot.core.data.reader.data_reader.get_values_from_df', lambda frame: frame.values)

    spec = DataSpec()

    result = from_csv_tsv(str(csv_path), spec)

    assert result is spec
    assert result.features.shape == (1, 1)
    assert captured['file_path'] == str(csv_path)
    assert captured['delimiter'] == '\t'
    assert captured['possible_idx_keywords'] == ['idx', 'index', 'id', 'unnamed: 0']


@pytest.mark.unit
def test_data_spec_normalizes_aliases_and_merges_dataloader_defaults():
    """Check DataSpec post-init normalization for enum aliases and dataloader options."""
    spec = DataSpec(
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
    assert spec.embedding_strategy is None
    assert spec.dataloader_kwargs == {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0,
        'drop_last': False,
    }


@pytest.mark.unit
def test_data_spec_normalizes_empty_and_missing_index_fields_by_default_semantics():
    """Check that empty optional index references and missing index collections use canonical values."""
    spec = DataSpec(
        target_idx=[],
        ts_terms_idx=[],
        categorical_idx=None,
        numerical_idx=None,
    )

    assert spec.target_idx is None
    assert spec.ts_terms_idx is None
    assert spec.categorical_idx == []
    assert spec.numerical_idx == []


@pytest.mark.unit
def test_data_spec_recreation_preserves_normalized_values_without_sharing_kwargs():
    """Check that recreating DataSpec from normalized fields is stable.

    Mutable dataloader kwargs should be copied so later changes do not leak between
    DataSpec instances.
    """
    first = DataSpec(
        task='classification',
        data_type='table',
        state='fit',
        ts_orientation=None,
        embedding_strategy={'method': 'demo'},
        dataloader_kwargs={'shuffle': False},
    )

    second = DataSpec(
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
    assert second.dataloader_kwargs is not first.dataloader_kwargs


@pytest.mark.unit
def test_preprocess_data_autodetects_tabular_data_type_for_classification(monkeypatch):
    """Check that missing data_type is inferred as tabular for ordinary classification tasks."""
    Backend().set('cpu')
    creator = TensorDataCreator()
    creator.spec = DataSpec(
        features=np.array([[1, 2], [3, 4]]),
        data_type=None,
        task='classification',
    )

    monkeypatch.setattr(TensorDataCreator, 'obligatory_preprocess', lambda self: None)

    creator.preprocess_data()

    assert creator.spec.data_type.value == 'table'
    assert creator.spec.data_type == creator.spec.data_type.tabular


@pytest.mark.unit
def test_preprocess_data_autodetects_time_series_data_type_for_forecasting(monkeypatch):
    """Check that missing data_type is inferred as time series for forecasting tasks."""
    Backend().set('cpu')
    creator = TensorDataCreator()
    creator.spec = DataSpec(
        features=np.array([[1, 2], [3, 4]]),
        data_type=None,
        task=Task(TaskTypesEnum.ts_forecasting),
    )

    monkeypatch.setattr(TensorDataCreator, 'obligatory_preprocess', lambda self: None)

    creator.preprocess_data()

    assert creator.spec.data_type.value == 'time_series'
    assert creator.spec.data_type == creator.spec.data_type.ts


@pytest.mark.unit
def test_tensor_data_creator_builds_tensor_data_container_from_spec_fields():
    """Check that TensorDataCreator copies prepared DataSpec fields into the runtime container."""
    creator = TensorDataCreator()
    creator.spec = DataSpec(
        features=np.array([[1.0, 2.0]]),
        target=np.array([1.0]),
        target_idx=1,
        categorical_idx=[0],
        numerical_idx=[1],
        features_names=['feature', 'target'],
    )

    tensor_data = creator.to_tensor_data()

    assert isinstance(tensor_data, TensorData)
    assert tensor_data.features is creator.spec.features
    assert tensor_data.target is creator.spec.target
    assert tensor_data.target_idx == 1
    assert tensor_data.categorical_idx == [0]
    assert tensor_data.features_names == ['feature', 'target']
