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
        use_stats = True
        super().__init__("prediction", cache_dir, use_stats=use_stats,
                         stats_keys=['predictions_hit', 'predictions_total'])
        self._init_db()
        if use_stats:
            self._init_db_stats()
        # TODO: probably initialize pickler only once

    def add_prediction(self, uid: str, outputData: OutputData):
        """
        Stores predicted `outputData` in binary fromat in a SQL table.
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
        Retrieves the predicted `outputData` from binary format in a SQL table.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    query = f"SELECT id, prediction FROM {self._main_table} WHERE id = ?"
                    cur.execute(query, (uid,))
                    result = cur.fetchone()
                    if result:
                        result = pickle.loads(result[1])
                    if self.use_stats:
                        if result:
                            self._inc_eff(cur, 'predictions_hit')
                            self._inc_stats(cur, uid)
                        self._inc_eff(cur, 'predictions_total')
            return result
        except sqlite3.Error as e:
            print(f"SQLite get Error: {e}")
            return None

    def retrieve_stats(self):
        """
        Retrieves stats for all non-zero pipeline nodes.
        """
        try:
            if self.use_stats:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    with conn:
                        cur = conn.cursor()
                        query = f"SELECT id, retrieve_count FROM stats;"
                        cur.execute(query)
                        return cur.fetchall()
        except sqlite3.Error as e:
            print(f"SQLite get Error: {e}")
            return None

    def _inc_stats(self, cur, id):
        """
        Increases retrieve count for each pipeline node.
        """
        try:
            query = """
            INSERT INTO stats (id, retrieve_count)
            VALUES (?, COALESCE((SELECT retrieve_count FROM stats WHERE id = ?), 0) + 1)
            ON CONFLICT(id) DO UPDATE SET
            retrieve_count = retrieve_count + 1
            """
            cur.execute(query, (id, id,))
        except sqlite3.Error as e:
            print(f"SQLite get Error: {e}")
            return None

    def _init_db_stats(self):
        """
        Initializes DB stats table.
        """
        try:
            if self.use_stats:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    with conn:
                        cur = conn.cursor()
                        query = f"CREATE TABLE IF NOT EXISTS stats (id TEXT PRIMARY KEY, retrieve_count INTEGER DEFAULT 0);"
                        cur.execute(query)
        except sqlite3.Error as e:
            print(f"SQLite get Error: {e}")

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
