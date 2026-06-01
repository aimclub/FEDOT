import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.utils import CACHE_DIR


@dataclass(frozen=True)
class TensorDataCacheIndexRecord:
    input_hash: str
    output_hash: str
    operation_hash: str
    path: Path
    created_at: str


@dataclass(frozen=True)
class PreprocessingModelCacheIndexRecord:
    model_hash: str
    operation_hash: str
    input_hash: str
    path: Path
    created_at: str


class CacheIndexDB:
    """
    SQLite index for cache files stored on disk.

    The index stores only metadata and paths. Cached objects themselves are
    written by `Saver` and loaded by `Loader`.
    """

    TENSOR_DATA_TABLE = "tensor_data_cache"
    PREPROCESSING_MODELS_TABLE = "preprocessing_model_cache"

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        ensure_cache_dirs()
        self.db_path = Path(db_path) if db_path is not None else CACHE_DIR / "index.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_tensor_data(
        self,
        input_hash: str,
        output_hash: str,
        operation_hash: str,
        path: Union[str, Path],
        created_at: Optional[str] = None,
    ) -> TensorDataCacheIndexRecord:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {self.TENSOR_DATA_TABLE}
                        (input_hash, output_hash, operation_hash, path, created_at)
                    VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    ON CONFLICT(input_hash, operation_hash) DO UPDATE SET
                        output_hash = excluded.output_hash,
                        path = excluded.path,
                        created_at = excluded.created_at;
                    """,
                    (input_hash, output_hash, operation_hash, str(path), created_at),
                )

        record = self.get_tensor_data(input_hash, operation_hash)
        if record is None:
            raise RuntimeError("TensorData cache index record was not saved.")
        return record

    def get_tensor_data(
        self,
        input_hash: str,
        operation_hash: str,
    ) -> Optional[TensorDataCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT input_hash, output_hash, operation_hash, path, created_at
                FROM {self.TENSOR_DATA_TABLE}
                WHERE input_hash = ? AND operation_hash = ?;
                """,
                (input_hash, operation_hash),
            )
            row = cur.fetchone()

        return self._tensor_data_record_from_row(row)

    def get_tensor_data_by_output_hash(
        self,
        output_hash: str,
    ) -> Optional[TensorDataCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT input_hash, output_hash, operation_hash, path, created_at
                FROM {self.TENSOR_DATA_TABLE}
                WHERE output_hash = ?;
                """,
                (output_hash,),
            )
            row = cur.fetchone()

        return self._tensor_data_record_from_row(row)

    def has_tensor_data(self, input_hash: str, operation_hash: str) -> bool:
        return self.get_tensor_data(input_hash, operation_hash) is not None

    def add_preprocessing_model(
        self,
        model_hash: str,
        operation_hash: str,
        input_hash: str,
        path: Union[str, Path],
        created_at: Optional[str] = None,
    ) -> PreprocessingModelCacheIndexRecord:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {self.PREPROCESSING_MODELS_TABLE}
                        (model_hash, operation_hash, input_hash, path, created_at)
                    VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    ON CONFLICT(input_hash, operation_hash) DO UPDATE SET
                        model_hash = excluded.model_hash,
                        path = excluded.path,
                        created_at = excluded.created_at;
                    """,
                    (model_hash, operation_hash, input_hash, str(path), created_at),
                )

        record = self.get_preprocessing_model(input_hash, operation_hash)
        if record is None:
            raise RuntimeError("Preprocessing model cache index record was not saved.")
        return record

    def get_preprocessing_model(
        self,
        input_hash: str,
        operation_hash: str,
    ) -> Optional[PreprocessingModelCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT model_hash, operation_hash, input_hash, path, created_at
                FROM {self.PREPROCESSING_MODELS_TABLE}
                WHERE input_hash = ? AND operation_hash = ?;
                """,
                (input_hash, operation_hash),
            )
            row = cur.fetchone()

        return self._preprocessing_model_record_from_row(row)

    def get_preprocessing_model_by_model_hash(
        self,
        model_hash: str,
    ) -> Optional[PreprocessingModelCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT model_hash, operation_hash, input_hash, path, created_at
                FROM {self.PREPROCESSING_MODELS_TABLE}
                WHERE model_hash = ?;
                """,
                (model_hash,),
            )
            row = cur.fetchone()

        return self._preprocessing_model_record_from_row(row)

    def has_preprocessing_model(self, input_hash: str, operation_hash: str) -> bool:
        return self.get_preprocessing_model(input_hash, operation_hash) is not None

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TENSOR_DATA_TABLE} (
                        input_hash TEXT NOT NULL,
                        output_hash TEXT NOT NULL,
                        operation_hash TEXT NOT NULL,
                        path TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (input_hash, operation_hash)
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.TENSOR_DATA_TABLE}_output_hash
                    ON {self.TENSOR_DATA_TABLE} (output_hash);
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.PREPROCESSING_MODELS_TABLE} (
                        model_hash TEXT NOT NULL,
                        operation_hash TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        path TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (input_hash, operation_hash)
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.PREPROCESSING_MODELS_TABLE}_model_hash
                    ON {self.PREPROCESSING_MODELS_TABLE} (model_hash);
                    """
                )

    @staticmethod
    def _tensor_data_record_from_row(row: Optional[tuple]) -> Optional[TensorDataCacheIndexRecord]:
        if row is None:
            return None

        input_hash, output_hash, operation_hash, path, created_at = row
        return TensorDataCacheIndexRecord(
            input_hash=input_hash,
            output_hash=output_hash,
            operation_hash=operation_hash,
            path=Path(path),
            created_at=created_at,
        )

    @staticmethod
    def _preprocessing_model_record_from_row(row: Optional[tuple]) -> Optional[PreprocessingModelCacheIndexRecord]:
        if row is None:
            return None

        model_hash, operation_hash, input_hash, path, created_at = row
        return PreprocessingModelCacheIndexRecord(
            model_hash=model_hash,
            operation_hash=operation_hash,
            input_hash=input_hash,
            path=Path(path),
            created_at=created_at,
        )
