import hashlib
import logging
import os
import timeit

import pandas as pd

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH


class DataCacher:
    """Class responsible for caching data of ``pd.DataFrame`` type in pickle format.
    Args:
        data_type_prefix: a string prefix related to the data to be cached. For example, if data is related to
        modelling results, then the prefix can be 'ModellingResults'. Default prefix is 'Data'.
        cache_folder: path to the folder where data is going to be cached.
    Examples:
        >>> your_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> data_cacher = DataCacher(data_type_prefix='data', cache_folder='your_path')
        >>> hashed_info = data_cacher.hash_info(dict(name='data', data=your_data))
        >>> data_cacher.cache_data(hashed_info, your_data)
        >>> data_cacher.load_data_from_cache(hashed_info)
    """

    def __init__(
            self,
            data_type_prefix: str = 'Data',
            cache_folder: str = None):
        self.data_type = data_type_prefix
        self.cache_folder = self._init_cache_folder(cache_folder)

        self.logger = logging.getLogger('DataCacher')

    def _init_cache_folder(self, cache_folder):
        if cache_folder is None:
            cache_folder = os.path.join(PROJECT_PATH, 'cache')
        else:
            cache_folder = os.path.abspath(cache_folder)

        os.makedirs(cache_folder, exist_ok=True)
        return cache_folder

    def hash_info(self, operation_info: dict) -> str:
        """Method responsible for hashing distinct information about the data that is going to be cached.
        It utilizes md5 hashing algorithm.
        Args:
            kwargs: a set of keyword arguments to be used as distinct info about data.
        Returns:
            Hashed string.
        """

        key = operation_info.encode('utf8')
        hsh = hashlib.md5(key).hexdigest()[:20]

        return hsh

    def load_data_from_cache(self, hashed_info: str):
        """Method responsible for loading cached data.
        Args:
            hashed_info: hashed string of needed info about the data.
        """
        self.logger.info('Trying to load features from cache')

        start = timeit.default_timer()
        file_path = os.path.join(self.cache_folder, hashed_info + '.npy')
        try:
            data = np.load(file_path)
        except FileNotFoundError:
            self.logger.info('Cache not found')
            raise FileNotFoundError(f'File {file_path} was not found')
        elapsed_time = round(timeit.default_timer() - start, 5)
        print(
            f'{self.data_type} of {type(data)} type are loaded from cache in {elapsed_time} sec')
        return data

    def cache_data(self, hashed_info: str, data: pd.DataFrame):
        """Method responsible for saving cached data. It utilizes pickle format for saving data.
        Args:
            hashed_info: hashed string.
            data: pd.DataFrame.
        """
        self.logger.info('Caching features')
        cache_file = os.path.join(self.cache_folder, hashed_info)

        try:
            np.save(cache_file, data)

        except Exception as ex:
            print(f'Data was not cached due to error { ex }')
