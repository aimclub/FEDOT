import pytest

from fedot.core.data.tensordata import TensorData
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
