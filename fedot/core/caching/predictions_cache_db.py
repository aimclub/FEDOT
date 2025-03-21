import pickle
import sqlite3
from contextlib import closing
from typing import Optional, List, Tuple

from fedot.core.caching.base_cache_db import BaseCacheDB
from fedot.core.data.data import OutputData


class PredictionsCacheDB(BaseCacheDB):
    """
    Database for `PredictionsCache` class.
    Includes low-level idea of caching predicted output using relational database.

    :param cache_dir: path to the place where cache files should be stored.
    """

    CREATE_STATS_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS stats (
        id TEXT PRIMARY KEY,
        retrieve_count INTEGER DEFAULT 0
    );
    """

    CREATE_MAIN_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        prediction BLOB
    );
    """

    INSERT_STATS_QUERY = """
    INSERT INTO stats (id, retrieve_count)
    VALUES (?, COALESCE((SELECT retrieve_count FROM stats WHERE id = ?), 0) + 1)
    ON CONFLICT(id) DO UPDATE SET
    retrieve_count = retrieve_count + 1
    """

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None):
        use_stats = True
        super().__init__("predictions",
                         cache_dir,
                         use_stats=use_stats,
                         stats_keys=['fit_hit', 'fit_total', 'pred_hit', 'pred_total'],
                         custom_pid=custom_pid)
        self._init_db()
        if use_stats:
            self._init_db_stats()

    def add_prediction(self, uid: str, type: str, outputData: OutputData) -> None:
        """
        Stores predicted `outputData` in binary format in a SQL table.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    pickled_data = sqlite3.Binary(pickle.dumps(outputData, pickle.HIGHEST_PROTOCOL))
                    cur.execute(f"INSERT OR IGNORE INTO {self._main_table} VALUES (?, ?);", [uid, pickled_data])
        except sqlite3.Error as e:
            self.log.error(f"SQLite add {type} error: {e}")

    def get_prediction(self, uid: str, type: str) -> Optional[OutputData]:
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
                            self._inc_eff(cur, f'{type}_hit')
                            self._inc_stats(cur, uid)
                        self._inc_eff(cur, f'{type}_total')
            return result
        except sqlite3.Error as e:
            self.log.error(f"SQLite get {type} Error: {e}")
            return None

    def retrieve_stats(self) -> Optional[List[Tuple[str, int]]]:
        """
        Retrieves statistics for all non-zero prediction cache entries.
        """
        try:
            if self.use_stats:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    with conn:
                        cur = conn.cursor()
                        query = "SELECT id, retrieve_count FROM stats;"
                        cur.execute(query)
                        return cur.fetchall()
        except sqlite3.Error as e:
            self.log.error(f"SQLite get Error: {e}")
            return None

    def _inc_stats(self, cur: sqlite3.Cursor, id: str) -> None:
        """
        Increases the retrieve count for a specific prediction cache entry.
        """
        try:
            cur.execute(self.INSERT_STATS_QUERY, (id, id,))
        except sqlite3.Error as e:
            self.log.error(f"SQLite inc stats Error: {e}")

    def _init_db_stats(self) -> None:
        """
        Initializes the database statistics table.
        """
        try:
            if self.use_stats:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    with conn:
                        cur = conn.cursor()
                        cur.execute(self.CREATE_STATS_TABLE_QUERY)
        except sqlite3.Error as e:
            self.log.error(f"SQLite init db stats Error: {e}")

    def _init_db(self) -> None:
        """
        Initializes the main database working table.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    cur.execute(self.CREATE_MAIN_TABLE_QUERY.format(table_name=self._main_table))
        except sqlite3.Error as e:
            self.log.error(f"SQLite init db error: {e}")
