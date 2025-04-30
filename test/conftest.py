from __future__ import annotations

from uuid import uuid4

import pytest

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.caching.predictions_cache import PredictionsCache
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
    if PredictionsCache in PredictionsCache._instances:
        del PredictionsCache._instances[PredictionsCache]

    unique_id_for_dbs = str(uuid4()).replace('-', '')
    use_stats = True

    OperationsCache(custom_pid=unique_id_for_dbs, use_stats=use_stats)
    PreprocessingCache(custom_pid=unique_id_for_dbs, use_stats=use_stats)
    PredictionsCache(custom_pid=unique_id_for_dbs, use_stats=use_stats)
    yield
