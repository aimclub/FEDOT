import numpy as np
import pandas as pd
import pytest
import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.reader.data_reader import DataReader, DataReaderResult
from fedot.core.data.reader.tools import read_arff_file
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.rules import TensorDataCreatorNotFoundError


class _FakeMeta:
    def __init__(self, names):
        self._names = list(names)

    def names(self):
        return list(self._names)


@pytest.mark.unit
def test_read_arff_file_returns_features_and_field_names(monkeypatch):
    fake_record = {
        'f0': np.array([1.0, 2.0, 3.0]),
        'f1': np.array([4.0, 5.0, 6.0]),
    }
    meta = _FakeMeta(['f0', 'f1'])

    monkeypatch.setattr('fedot.core.data.reader.tools.loadarff',
                        lambda _path: (fake_record, meta))
    monkeypatch.setattr(Backend(), 'name', 'cpu')
    monkeypatch.setattr(Backend(), 'xp', np)

    features, field_names = read_arff_file('sample.arff')

    assert features.shape == (2, 3)
    assert np.allclose(features[0], [1.0, 2.0, 3.0])
    assert np.allclose(features[1], [4.0, 5.0, 6.0])
    assert field_names == ['f0', 'f1']


@pytest.mark.unit
def test_read_arff_file_empty_attribute_list_sets_field_names_to_none(monkeypatch):
    fake_record = {}
    meta = _FakeMeta([])

    monkeypatch.setattr('fedot.core.data.reader.tools.loadarff',
                        lambda _path: (fake_record, meta))
    monkeypatch.setattr(Backend(), 'name', 'cpu')
    monkeypatch.setattr(Backend(), 'xp', np)

    features, field_names = read_arff_file('empty.arff')

    assert field_names is None
    assert features.shape == (0,)


@pytest.mark.unit
def test_read_arff_file_minimal_arff_roundtrip(tmp_path):
    arff = b"""@relation tiny
@attribute x numeric
@attribute y numeric
@data
0,1
2,3
"""
    path = tmp_path / 'tiny.arff'
    path.write_bytes(arff)
    Backend().set('cpu')

    features, field_names = read_arff_file(str(path))

    assert field_names == ['x', 'y']
    assert features.shape == (2, 2)
    assert np.allclose(features, [[0.0, 2.0], [1.0, 3.0]])


@pytest.mark.unit
def test_data_reader_read_csv_returns_features_and_column_names(tmp_path):
    path = tmp_path / 'data.csv'
    pd.DataFrame({'c0': [1, 2], 'c1': [3, 4]}).to_csv(path, index=False)
    Backend().set('cpu')

    out = DataReader.read(str(path), DataSpec())

    assert isinstance(out, DataReaderResult)
    assert out.features.shape == (2, 2)
    assert np.asarray(out.features_names).tolist() == ['c0', 'c1']


@pytest.mark.unit
def test_data_reader_read_numpy_returns_same_array():
    Backend().set('cpu')
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    out = DataReader.read(arr, DataSpec())

    assert isinstance(out, DataReaderResult)
    assert out.features is arr
    assert out.features_names is None


@pytest.mark.unit
def test_data_reader_read_torch_tensor_returns_same_tensor():
    Backend().set('cpu')
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    out = DataReader.read(t, DataSpec())

    assert isinstance(out, DataReaderResult)
    assert out.features is t


@pytest.mark.unit
def test_data_reader_read_arff_uses_read_arff_file(tmp_path, monkeypatch):
    path = tmp_path / 'via_reader.arff'
    path.write_text('', encoding='ascii')

    def fake_read_arff(_source):
        return np.zeros((1, 2)), ['a', 'b']

    monkeypatch.setattr(
        'fedot.core.data.reader.data_reader.read_arff_file', fake_read_arff)

    out = DataReader.read(str(path), DataSpec())

    assert isinstance(out, DataReaderResult)
    assert out.features.shape == (1, 2)
    assert out.features_names == ['a', 'b']


@pytest.mark.unit
def test_data_reader_unknown_source_raises():
    Backend().set('cpu')

    with pytest.raises(TensorDataCreatorNotFoundError):
        DataReader.read({'not': 'supported'}, DataSpec())
