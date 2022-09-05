import pytest

from fedot.core.utilities.singleton_meta import SingletonMeta


@pytest.fixture
def cleanup_singletons():
    keys = list(SingletonMeta._instances)
    for inst in keys:
        del SingletonMeta._instances[inst]
        del inst
    yield
    keys = list(SingletonMeta._instances)
    for inst in keys:
        del SingletonMeta._instances[inst]
        del inst
