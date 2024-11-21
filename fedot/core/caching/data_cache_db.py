import pickle
import sqlite3
from contextlib import closing
from os import getpid
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from fedot.core.caching.base_cache_db import BaseCacheDB
from fedot.core.data.data import OutputData


class DataCacheDB(BaseCacheDB):
    """
    Database for `DataCache` class.
    Includes low-level idea of caching predicted output using relational database.

    :param cache_dir: path to the place where cache files should be stored.
    """

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None):
        super().__init__("prediction", cache_dir)
        self._init_db()
        # TODO: probably initialize pickler only once

    def add_prediction(self, uid: str, outputData: OutputData):
        """
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    pickled_data = sqlite3.Binary(pickle.dumps(outputData, pickle.HIGHEST_PROTOCOL))
                    cur.execute(f"INSERT OR IGNORE INTO {self._main_table} VALUES (?, ?);", [uid, pickled_data])
        except sqlite3.Error as e:
            print(f"SQLite add error: {e}")

    def get_prediction(self, uid: str) -> Optional[OutputData]:
        """
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    query = f"SELECT id, prediction FROM {self._main_table} WHERE id = ?"
                    cur.execute(query, (uid,))
                    result = cur.fetchone()
                    if result:
                        result = pickle.loads(result[0])
            return result
        except sqlite3.Error as e:
            print(f"SQLite get Error: {e}")
            return None

    def _init_db(self):
        """
        Initializes DB working table.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    cur.execute(
                        (
                            f"CREATE TABLE IF NOT EXISTS {self._main_table} ("
                            "id TEXT PRIMARY KEY,"
                            "prediction BLOB"
                            ");"
                        )
                    )
        except sqlite3.Error as e:
            print(f"SQLite init error: {e}")
