from uuid import uuid4

import pytest

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.utils import set_random_seed


@pytest.fixture(scope='session', autouse=True)
def establish_seed():
    set_random_seed(42)


@pytest.fixture(scope='function', autouse=True)
def run_around_tests():
    # remove singleton from previous run #TODO refactor
    if OperationsCache in OperationsCache._instances:
        del OperationsCache._instances[OperationsCache]
    if PreprocessingCache in PreprocessingCache._instances:
        del PreprocessingCache._instances[PreprocessingCache]

    unique_id_for_dbs = str(uuid4()).replace('-', '')

    OperationsCache(custom_pid=unique_id_for_dbs)
    PreprocessingCache(custom_pid=unique_id_for_dbs)
    yield