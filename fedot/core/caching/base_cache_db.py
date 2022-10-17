import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional, Sequence, Tuple

import psutil

from fedot.core.utils import default_fedot_data_dir


class BaseCacheDB:
    """
    Base class for caching in database.
    Includes low-level idea of caching data using relational database.

    :param main_table: table to store into or load from
    :param cache_folder: path to the place where cache files should be stored.
    :param use_stats: bool indicating if it is needed to use cache performance dict
    :param stats_keys: sequence of keys for supporting cache effectiveness
    """

    def __init__(self, main_table: str = 'default', cache_folder: Optional[str] = None, use_stats: bool = False,
                 stats_keys: Sequence = ('default_hit', 'default_total')):
        self._main_table = main_table
        self._db_suffix = f'.{main_table}_db'
        if cache_folder is None:
            self.db_path = Path(default_fedot_data_dir())
            self._del_prev_temps()
        else:
            self.db_path = Path(cache_folder)
        self.db_path = self.db_path.joinpath(f'cache_{os.getpid()}').with_suffix(self._db_suffix)

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
        Deletes previously generated unused DB files.
        """
        for file in self.db_path.glob(f'cache_*{self._db_suffix}'):
            try:
                pid = int(file.stem.split('_')[-1])
            except ValueError:
                pid = -1  # old format cache name, remove this line somewhere in the future
            if pid not in psutil.pids():
                try:
                    file.unlink()
                except FileNotFoundError:
                    pass  # it means another process have already killed it
                except PermissionError:
                    pass  # the same

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
        cur.execute(f'DELETE FROM {self._main_table};')

    def __len__(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT id FROM {self._main_table};')
                all_rows = cur.fetchall()
                return len(all_rows)
