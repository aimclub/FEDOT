import pytest

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.utils import set_random_seed


@pytest.fixture(scope='session', autouse=True)
def establish_seed():
    set_random_seed(42)


@pytest.fixture(scope='function', autouse=True)
def run_around_tests():
    OperationsCache()
    PreprocessingCache()
    yield
    OperationsCache().reset(full_clean=True)
    PreprocessingCache().reset(full_clean=True)
