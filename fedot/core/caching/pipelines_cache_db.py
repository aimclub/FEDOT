import pickle
import sqlite3
from contextlib import closing
from os import getpid
from typing import List, Optional, Tuple, TypeVar

from fedot.core.caching.base_cache_db import BaseCacheDB
from fedot.core.operations.operation import Operation

IOperation = TypeVar('IOperation', bound=Operation)


class OperationsCacheDB(BaseCacheDB):
    """
    Database for `OperationsCache` class.
    Includes low-level idea of caching pipeline nodes using relational database.

    :param cache_folder: path to the place where cache files should be stored.
    """

    def __init__(self, cache_folder: Optional[str] = None):
        super().__init__('operations', cache_folder, False, ['pipelines_hit', 'pipelines_total', 'nodes_hit', 'nodes_total'])
        self._init_db()

    @staticmethod
    def _create_temp_for_ordered_select(cur: sqlite3.Cursor, uids: List[str]) -> str:
        """
        Creates temp table to keep order of uids while doing select operation in operations getter.

        :param cur: cursor with already installed DB connection
        :param uids: list of operations uids for keeping the order

        :return tmp_name: str name of newly created temp table
        """
        _, *other = uids
        tmp_name = f'tmp_{getpid()}'
        query = f'CREATE TEMP TABLE {tmp_name} AS SELECT 1 as id1, ? as id2'
        for num, _ in enumerate(other, 2):
            query += f' union SELECT {num} as id1, ? as id2'
        cur.execute(query, uids)
        return tmp_name

    def get_operations(self, uids: List[str]) -> List[Optional['IOperation']]:
        """
        Maps given uids to scores from DB and puts None if is not present.

        :param uids: list of operations uids to be mapped

        :return retrieved: list of operations taken from DB table with None where it wasn't present
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                tmp_name = self._create_temp_for_ordered_select(cur, uids)
                cur.execute((
                    f'SELECT operation FROM {tmp_name} '
                    f'LEFT JOIN {self._main_table} ON {self._main_table}.id = {tmp_name}.id2 '
                    f'ORDER BY {tmp_name}.id1;'
                ))
                retrieved = cur.fetchall()
                if self.use_stats:
                    non_null = [x for (x,) in retrieved if x is not None]
                    self._inc_eff(cur, 'nodes_hit', len(non_null))
                    if len(non_null) == len(uids):
                        self._inc_eff(cur, 'pipelines_hit')
                    self._inc_eff(cur, 'nodes_total', len(uids))
                    self._inc_eff(cur, 'pipelines_total')
                retrieved = [pickle.loads(x) if x is not None else None for (x,) in retrieved]
                return retrieved

    def add_operations(self, uid_val_lst: List[Tuple[str, 'IOperation']]):
        """
        Adds operation score to DB table via its uid

        :param uid_val_lst: list of pairs (uid -> operation) to be saved
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                pickled = [
                    (uid, sqlite3.Binary(pickle.dumps(val, pickle.HIGHEST_PROTOCOL)))
                    for uid, val in uid_val_lst
                ]
                cur.executemany(f'INSERT OR IGNORE INTO {self._main_table} VALUES (?, ?);', pickled)

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
                    'operation BLOB'
                    ');'
                ))
