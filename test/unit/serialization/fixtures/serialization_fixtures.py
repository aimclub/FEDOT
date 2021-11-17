import pytest
from fedot.serializers import json_helpers


@pytest.fixture
def _get_class_fixture(monkeypatch):
    def mock_get_class(obj_type: type):
        return obj_type

    monkeypatch.setattr(
        f'{json_helpers._get_class.__module__}.{json_helpers._get_class.__qualname__}',
        mock_get_class
    )
