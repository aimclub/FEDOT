import numpy as np
import pytest

from fedot.core.backend.backend import backend
from fedot.core.data.reader.tools import read_arff_file


class _FakeMeta:
    def names(self):
        return ['feature', 'target']


@pytest.mark.unit
def test_read_arff_file_uses_target_resolution_rules(monkeypatch):
    fake_record = {
        'feature': np.array([1.0, 2.0]),
        'target': np.array([b'a', b'b']),
    }

    monkeypatch.setattr('fedot.core.data.data_reader.loadarff', lambda _path: (fake_record, _FakeMeta()))
    monkeypatch.setattr(backend, 'name', 'cpu')
    monkeypatch.setattr(backend, 'xp', np)

    features, target = read_arff_file('sample.arff', target_idx='target')

    assert features.shape == (1, 2)
    assert target.tolist() == ['a', 'b']


@pytest.mark.unit
def test_read_arff_file_infers_target_when_not_provided(monkeypatch):
    fake_record = {
        'feature': np.array([1.0, 2.0]),
        'target': np.array([b'a', b'b']),
    }

    monkeypatch.setattr('fedot.core.data.data_reader.loadarff', lambda _path: (fake_record, _FakeMeta()))
    monkeypatch.setattr(backend, 'name', 'cpu')
    monkeypatch.setattr(backend, 'xp', np)

    features, target = read_arff_file('sample.arff')

    assert features.shape == (1, 2)
    assert target.tolist() == ['a', 'b']
