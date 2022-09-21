import pickle
import sqlite3
from contextlib import closing
from typing import Dict, Optional, Tuple

from fedot.core.caching.base_cache_db import BaseCacheDB
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)
from fedot.preprocessing.preprocessing import DataPreprocessor


class PreprocessingCacheDB(BaseCacheDB):
    """
    Database for `PreprocessingCache` class.
    Includes low-level idea of caching pipeline preprocessor items using relational database.

    :param cache_folder: path to the place where cache files should be stored.
    """

    def __init__(self, cache_folder: Optional[str] = None):
        super().__init__('preprocessors', cache_folder, False, ['preprocessors_hit', 'preprocessors_total'])
        self._init_db()

    def get_preprocessor(self, uid: str) -> Optional[Tuple[
        Dict[str, OneHotEncodingImplementation], Dict[str, ImputationImplementation]
    ]]:
        """
        Tries to return both data processors, None if is not found

        :param uid: uid of preproccesor to be loaded

        :return matched: pair of data processors (encoder, imputer) or None
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT encoder, imputer FROM {self._main_table} WHERE id = ?;', [uid])
                matched = cur.fetchone()
                is_loaded = False
                if matched is not None:
                    matched = tuple([pickle.loads(matched[i]) for i in range(2)])
                    is_loaded = True
                if self.use_stats:
                    if is_loaded:
                        self._inc_eff(cur, 'preprocessors_hit')
                    self._inc_eff(cur, 'preprocessors_total')
                return matched

    def add_preprocessor(self, uid: str, value: DataPreprocessor):
        """
        Adds preprocessor score to DB table vid its uid.

        :param uid: unique preprocessor identificator
        :param value: the preprocessor itself
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                pickled_encoder = sqlite3.Binary(pickle.dumps(value.features_encoders, pickle.HIGHEST_PROTOCOL))
                pickled_imputer = sqlite3.Binary(pickle.dumps(value.features_imputers, pickle.HIGHEST_PROTOCOL))
                cur.execute(f'INSERT OR IGNORE INTO {self._main_table} VALUES (?, ?, ?);',
                            [uid, pickled_encoder, pickled_imputer])

    def _init_db(self):
        """
        Initializes DB working table.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._main_table} ('
                    'id TEXT PRIMARY KEY,'
                    'encoder BLOB,'
                    'imputer BLOB'
                    ');'
                ))
