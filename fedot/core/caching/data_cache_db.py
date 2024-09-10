import pickle
import sqlite3
from contextlib import closing
from os import getpid
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from fedot.core.caching.base_cache_db import BaseCacheDB


class DataCacheDB(BaseCacheDB):
    """
    Database for `DataCache` class.
    Includes low-level idea of caching predicted output using relational database.

    :param cache_dir: path to the place where cache files should be stored.
    """

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None):
        super().__init__("prediction", cache_dir)
        self._init_db()

    def add_prediction(self, uid_val_lst: List[Tuple[str, np.ndarray]]):
        """
        Adds operation score to DB table via its uid

        :param uid_val_lst: list of pairs (uid -> prediction) to be saved
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    pickled = [
                        (
                            uid,
                            sqlite3.Binary(pickle.dumps(val, pickle.HIGHEST_PROTOCOL)),
                        )
                        for uid, val in uid_val_lst
                    ]
                    cur.executemany(
                        f"INSERT OR IGNORE INTO {self._main_table} VALUES (?, ?);",
                        pickled,
                    )
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")

    def get_prediction(self, uids: List[str]) -> List[Optional[np.ndarray]]:
        """
        Maps given uids to operations from DB and puts None if is not present.

        :param uids: list of operations uids to be mapped

        :return retrieved: list of operations taken from DB table with None where it wasn't present
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    placeholders = ",".join("?" for _ in uids)
                    query = (
                        f"SELECT id, prediction FROM {self._main_table} "
                        f"WHERE id IN ({placeholders})"
                    )
                    cur.execute(query, uids)
                    results = {row[0]: pickle.loads(row[1]) for row in cur.fetchall()}
                    retrieved = [results.get(uid) for uid in uids]
            return retrieved
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return [None] * len(uids)

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
            print(f"SQLite error: {e}")
