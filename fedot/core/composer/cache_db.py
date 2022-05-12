import pickle
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from fedot.core.utils import default_fedot_data_dir

if TYPE_CHECKING:
    from .cache import CachedState


class OperationsCacheDB:
    """
    Database for OperationsCache class.
    Includes low-level idea of caching pipeline nodes using sqlite3.

    :param db_path: str db file path
    """

    def __init__(self, db_path: str):
        self.db_path = db_path or Path(default_fedot_data_dir(), f'tmp_{str(uuid.uuid4())}')
        self._db_suffix = '.cache_db'
        self.db_path = Path(self.db_path).with_suffix(self._db_suffix)

        self._del_prev_temps()

        self._effectiveness_keys = ['pipelines_hit', 'nodes_hit', 'pipelines_total', 'nodes_total']
        self._eff_table = 'effectiveness'
        self._op_table = 'operations'
        self._init_db()

    def get_effectiveness(self) -> Tuple[int, int, int, int]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT {",".join(self._effectiveness_keys)} FROM {self._eff_table};')
                return cur.fetchone()

    def reset(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                self._reset_eff(cur)
                self._reset_ops(cur)

    def _del_prev_temps(self):
        for file in self.db_path.parent.glob(f'tmp_*{self._db_suffix}'):
            file.unlink()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                eff_type = ' INTEGER DEFAULT 0'
                fields = f'{eff_type},'.join(self._effectiveness_keys) + eff_type
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._eff_table} ('
                    'id INTEGER PRIMARY KEY CHECK (id = 1),'  # noqa better viewed like that
                    f'{fields}'  # noqa
                    ');'
                ))
                cur.execute(f'INSERT INTO {self._eff_table} DEFAULT VALUES;')
            with conn:
                cur = conn.cursor()
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._op_table} ('
                    'id TEXT PRIMARY KEY,'  # noqa better viewed like that
                    'operation BLOB'  # noqa
                    ');'
                ))

    def _inc_eff(self, cur: sqlite3.Cursor, col: str, inc_val: int = 1):
        cur.execute(f'UPDATE {self._eff_table} SET {col} = {col} + {inc_val};')

    def _reset_eff(self, cur: sqlite3.Cursor):
        cur.execute(f'DELETE FROM {self._eff_table};')
        cur.execute(f'INSERT INTO {self._eff_table} DEFAULT VALUES;')

    def _reset_ops(self, cur: sqlite3.Cursor):
        cur.execute(f'DELETE FROM {self._op_table};')

    @staticmethod
    def _create_temp_for_ordered_select(cur: sqlite3.Cursor, uids: List[str]):
        _, *other = uids
        tmp_name = 'tmp'
        cur.execute(f'DROP TABLE IF EXISTS {tmp_name};')  # TODO: make truly temp table, not like that
        query = (
            f'CREATE TABLE {tmp_name} AS '
            'SELECT 1 as id1, ? as id2'
        )
        for num, _ in enumerate(other, 2):
            query += (
                ' union '
                f'SELECT {num} as id1, ? as id2'
            )
        cur.execute(query, uids)
        return tmp_name

    def get_operations(self, uids: List[str]) -> List[Optional['CachedState']]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                tmp_name = self._create_temp_for_ordered_select(cur, uids)
                cur.execute((
                    f'SELECT operation FROM {tmp_name} '
                    f'LEFT JOIN {self._op_table} ON {self._op_table}.id = {tmp_name}.id2 '
                    f'ORDER BY {tmp_name}.id1;'
                ))
                retrieved = cur.fetchall()
                non_null = [x for (x,) in retrieved if x is not None]
                self._inc_eff(cur, 'nodes_hit', len(non_null))
                if len(non_null) == len(uids):
                    self._inc_eff(cur, 'pipelines_hit')
                retrieved = [pickle.loads(x) if x is not None else None for (x,) in retrieved]
                self._inc_eff(cur, 'nodes_total', len(uids))
                self._inc_eff(cur, 'pipelines_total')
                return retrieved

    def add_operation(self, conn: sqlite3.Connection, uid: str, val: 'CachedState'):
        with conn:
            cur = conn.cursor()
            pdata = pickle.dumps(val, pickle.HIGHEST_PROTOCOL)
            cur.execute(f'INSERT OR IGNORE INTO {self._op_table} VALUES (?, ?);',
                        [uid, sqlite3.Binary(pdata)])

    def add_operations(self, uid_val_lst: List[Tuple[str, 'CachedState']]):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                pickled = [
                    (uid, sqlite3.Binary(pickle.dumps(val, pickle.HIGHEST_PROTOCOL)))
                    for uid, val in uid_val_lst
                ]
                cur.executemany(f'INSERT OR IGNORE INTO {self._op_table} VALUES (?, ?);', pickled)

    def __len__(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT id FROM {self._op_table};')
                all_rows = cur.fetchall()
                return len(all_rows)
