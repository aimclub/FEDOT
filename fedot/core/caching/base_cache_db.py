import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Optional, Sequence, Tuple

from fedot.core.utils import default_fedot_data_dir


class BaseCacheDB:
    """
    Base class for caching in database.
    Includes low-level idea of caching data using relational database.

    :param main_table: table to store into or load from
    :param db_path: str db file path
    :param use_stats: bool indicating if it is needed to use cache performance dict
    :param stats_keys: sequence of keys for supporting cache effectiveness
    """

    def __init__(self, main_table: str = 'default', db_path: Optional[str] = None, use_stats: bool = False,
                 stats_keys: Sequence = ('default_hit', 'default_total')):
        self._main_table = main_table
        self._db_suffix = f'.{main_table}_db'
        self.db_path = db_path or Path(default_fedot_data_dir(), f'cache_{str(uuid.uuid4())}')
        self.db_path = Path(self.db_path).with_suffix(self._db_suffix)

        self._del_prev_temps()

        self._eff_table = 'effectiveness'
        self.use_stats = use_stats
        self._effectiveness_keys = stats_keys
        self._init_eff()

    def get_effectiveness(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns effectiveness of the cache in case of enabled `use_stats`, None instead.
        """
        if self.use_stats:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    cur.execute(f'SELECT {",".join(self._effectiveness_keys)} FROM {self._eff_table};')
                    return cur.fetchone()

    def get_effectiveness_keys(self) -> Sequence:
        """
        Returns all cache effectiveness keys.
        """
        return self._effectiveness_keys

    def reset(self):
        """
        Drops all scores from working table and resets efficiency table values to zero.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                if self.use_stats:
                    self._reset_eff(cur)
                self._reset_main(cur)

    def _init_eff(self):
        """
        Initializes effectiveness table.
        """
        if self.use_stats:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cur = conn.cursor()
                    eff_type = ' INTEGER DEFAULT 0'
                    fields = f'{eff_type},'.join(self._effectiveness_keys) + eff_type
                    cur.execute((
                        f'CREATE TABLE IF NOT EXISTS {self._eff_table} ('
                        'id INTEGER PRIMARY KEY CHECK (id = 1),'
                        f'{fields}'
                        ');'
                    ))
                    cur.execute(f'INSERT OR IGNORE INTO {self._eff_table} DEFAULT VALUES;')

    def _del_prev_temps(self):
        """
        Deletes previously generated DB files.
        """
        for file in self.db_path.parent.glob(f'cache_*{self._db_suffix}'):
            file.unlink()

    def _inc_eff(self, cur: sqlite3.Cursor, col: str, inc_val: int = 1):
        """
        Increases `col` score in efficiency table by `inc_val`.

        :param cur: cursor with already installed DB connection
        :param col: column of efficiency table to increase
        :param inc_val: value to increase column
        """
        cur.execute(f'UPDATE {self._eff_table} SET {col} = {col} + {inc_val};')

    def _reset_eff(self, cur: sqlite3.Cursor):
        """
        Resets efficiency table scores to zero.

        :param cur: cursor with already installed DB connection
        """
        cur.execute(f'DELETE FROM {self._eff_table};')
        cur.execute(f'INSERT INTO {self._eff_table} DEFAULT VALUES;')

    def _reset_main(self, cur: sqlite3.Cursor):
        """
        Drops all scores from working table.

        :param cur: cursor with already installed DB connection
        """
        try:
            cur.execute(f'DELETE FROM {self._main_table};')
        except:
            pass

    def __len__(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT id FROM {self._main_table};')
                all_rows = cur.fetchall()
                return len(all_rows)
