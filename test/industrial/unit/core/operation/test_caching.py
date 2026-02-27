import os
import pytest
import pandas as pd

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.caching import DataCacher
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


@pytest.fixture
def cache_folder():
    return os.path.join(PROJECT_PATH, 'cache')


@pytest.fixture
def data_cacher(cache_folder):
    return DataCacher(data_type_prefix='data', cache_folder=cache_folder)


@pytest.fixture
def test_data():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})


@pytest.fixture
def cache_info(test_data):
    return str({'name': test_data, 'add_info': ['fedot', 'industrial']})


def test_hash_info(data_cacher, cache_info):
    hashed_info = data_cacher.hash_info(operation_info=cache_info)
    assert isinstance(hashed_info, str)
    assert len(hashed_info) == 20


def test_cache_data(data_cacher, test_data, cache_info):
    hashed_info = data_cacher.hash_info(operation_info=cache_info)
    data_cacher.cache_data(hashed_info, test_data)
    cache_file = os.path.join(data_cacher.cache_folder, hashed_info + '.npy')

    assert os.path.isfile(cache_file)


def test_load_data_from_cache(data_cacher, test_data, cache_info):
    hashed_info = data_cacher.hash_info(operation_info=cache_info)
    data_cacher.cache_data(hashed_info, test_data)
    loaded_data = data_cacher.load_data_from_cache(hashed_info)

    assert isinstance(loaded_data, np.ndarray)
    assert (test_data.values == loaded_data).all()
