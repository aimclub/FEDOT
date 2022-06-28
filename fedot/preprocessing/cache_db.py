import pickle
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Optional, Tuple

from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)
from fedot.core.utils import default_fedot_data_dir
from fedot.preprocessing.preprocessing import DataPreprocessor


class PreprocessingCacheDB:
    """
    Database for PreprocessingCache class.
    Includes low-level idea of caching pipeline preprocessor using sqlite3.

    :param db_path: str db file path
    """

    def __init__(self, db_path: Optional[str] = None):
        self._preproc_table = 'preprocessors'
        self._db_suffix = '.preprocessing_db'
        self.db_path = db_path or Path(default_fedot_data_dir(), f'prp_{str(uuid.uuid4())}')
        self.db_path = Path(self.db_path).with_suffix(self._db_suffix)

        self._del_prev_temps()
        self._init_db()

    def get_preprocessor(self, uid: str) -> Tuple[
        Optional[OneHotEncodingImplementation], Optional[ImputationImplementation]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'SELECT encoder, imputer FROM {self._preproc_table} WHERE id = ?;', [uid])
                matched = cur.fetchone()
                if matched is not None:
                    matched = tuple([pickle.loads(matched[i]) for i in range(2)])
                else:
                    matched = None, None
                return matched

    def add_preprocessor(self, uid: str, value: DataPreprocessor):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                pickled_encoder = sqlite3.Binary(pickle.dumps(value.features_encoders, pickle.HIGHEST_PROTOCOL))
                pickled_imputer = sqlite3.Binary(pickle.dumps(value.features_imputers, pickle.HIGHEST_PROTOCOL))
                cur.execute(f'INSERT OR IGNORE INTO {self._preproc_table} VALUES (?, ?, ?);',
                            [uid, pickled_encoder, pickled_imputer])

    def reset(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(f'DELETE FROM {self._preproc_table};')

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute((
                    f'CREATE TABLE IF NOT EXISTS {self._preproc_table} ('
                    'id TEXT PRIMARY KEY,'
                    'encoder BLOB,'
                    'imputer BLOB'
                    ');'
                ))

    def _del_prev_temps(self):
        for file in self.db_path.parent.glob(f'prp_*{self._db_suffix}'):
            file.unlink()
