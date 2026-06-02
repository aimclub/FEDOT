import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from fedot.core.caching.normalization import normalize_for_hash
from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.utils import CACHE_DIR


@dataclass(frozen=True)
class TensorDataCacheIndexRecord:
    input_hash: str
    output_hash: str
    operation_hash: str
    state: str
    path: Path
    created_at: str


@dataclass(frozen=True)
class PreprocessingModelCacheIndexRecord:
    model_hash: str
    operation_hash: str
    input_hash: str
    path: Path
    created_at: str
    step_order: int = 0
    step_name: Optional[str] = None
    method: Optional[str] = None
    features_idx: Optional[Any] = None


@dataclass(frozen=True)
class PreprocessingPlanCacheIndexRecord:
    plan_hash: str
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
    PREPROCESSING_PLANS_TABLE = "preprocessing_plan_cache"

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
        state: str = "fit",
        created_at: Optional[str] = None,
    ) -> TensorDataCacheIndexRecord:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {self.TENSOR_DATA_TABLE}
                        (input_hash, output_hash, operation_hash, state, path, created_at)
                    VALUES (?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    ON CONFLICT(input_hash, operation_hash) DO UPDATE SET
                        output_hash = excluded.output_hash,
                        state = excluded.state,
                        path = excluded.path,
                        created_at = excluded.created_at;
                    """,
                    (input_hash, output_hash, operation_hash, state, str(path), created_at),
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
                SELECT input_hash, output_hash, operation_hash, state, path, created_at
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
                SELECT input_hash, output_hash, operation_hash, state, path, created_at
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
        step_order: int = 0,
        step_name: Optional[str] = None,
        method: Optional[str] = None,
        features_idx: Optional[Any] = None,
        created_at: Optional[str] = None,
    ) -> PreprocessingModelCacheIndexRecord:
        features_idx_json = self._features_idx_to_json(features_idx)
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {self.PREPROCESSING_MODELS_TABLE}
                        (
                            model_hash, operation_hash, input_hash, path,
                            step_order, step_name, method, features_idx, created_at
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    ON CONFLICT(input_hash, operation_hash, model_hash) DO UPDATE SET
                        path = excluded.path,
                        step_order = excluded.step_order,
                        step_name = excluded.step_name,
                        method = excluded.method,
                        features_idx = excluded.features_idx,
                        created_at = excluded.created_at;
                    """,
                    (
                        model_hash, operation_hash, input_hash, str(path),
                        step_order, step_name, method, features_idx_json, created_at,
                    ),
                )

        record = self.get_preprocessing_model_by_model_hash(model_hash)
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
                SELECT
                    model_hash, operation_hash, input_hash, path, created_at,
                    step_order, step_name, method, features_idx
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
                SELECT
                    model_hash, operation_hash, input_hash, path, created_at,
                    step_order, step_name, method, features_idx
                FROM {self.PREPROCESSING_MODELS_TABLE}
                WHERE model_hash = ?;
                """,
                (model_hash,),
            )
            row = cur.fetchone()

        return self._preprocessing_model_record_from_row(row)

    def get_preprocessing_models(
        self,
        input_hash: str,
        operation_hash: str,
    ) -> list[PreprocessingModelCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT
                    model_hash, operation_hash, input_hash, path, created_at,
                    step_order, step_name, method, features_idx
                FROM {self.PREPROCESSING_MODELS_TABLE}
                WHERE input_hash = ? AND operation_hash = ?
                ORDER BY step_order, created_at, model_hash;
                """,
                (input_hash, operation_hash),
            )
            rows = cur.fetchall()

        return [
            record for row in rows
            if (record := self._preprocessing_model_record_from_row(row)) is not None
        ]

    def has_preprocessing_model(self, input_hash: str, operation_hash: str) -> bool:
        return self.get_preprocessing_model(input_hash, operation_hash) is not None

    def add_preprocessing_plan(
        self,
        plan_hash: str,
        path: Union[str, Path],
        created_at: Optional[str] = None,
    ) -> PreprocessingPlanCacheIndexRecord:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {self.PREPROCESSING_PLANS_TABLE}
                        (plan_hash, path, created_at)
                    VALUES (?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    ON CONFLICT(plan_hash) DO UPDATE SET
                        path = excluded.path,
                        created_at = excluded.created_at;
                    """,
                    (plan_hash, str(path), created_at),
                )

        record = self.get_preprocessing_plan(plan_hash)
        if record is None:
            raise RuntimeError("Preprocessing plan cache index record was not saved.")
        return record
    
    def get_preprocessing_plan(self, plan_hash: str) -> Optional[PreprocessingPlanCacheIndexRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT plan_hash, path, created_at FROM {self.PREPROCESSING_PLANS_TABLE} WHERE plan_hash = ?;
                """,
                (plan_hash,),
            )
            row = cur.fetchone()

        return self._preprocessing_plan_record_from_row(row)

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
                        state TEXT NOT NULL,
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
                        step_order INTEGER NOT NULL DEFAULT 0,
                        step_name TEXT,
                        method TEXT,
                        features_idx TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (input_hash, operation_hash, model_hash)
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.PREPROCESSING_MODELS_TABLE}_model_hash
                    ON {self.PREPROCESSING_MODELS_TABLE} (model_hash);
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.PREPROCESSING_PLANS_TABLE} (
                        plan_hash TEXT NOT NULL,
                        path TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (plan_hash)
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.PREPROCESSING_PLANS_TABLE}_plan_hash
                    ON {self.PREPROCESSING_PLANS_TABLE} (plan_hash);
                    """
                )
                self._ensure_tensor_table_schema(cur)
                self._ensure_preprocessing_models_table_schema(cur)

    @staticmethod
    def _tensor_data_record_from_row(row: Optional[tuple]) -> Optional[TensorDataCacheIndexRecord]:
        if row is None:
            return None

        input_hash, output_hash, operation_hash, state, path, created_at = row
        return TensorDataCacheIndexRecord(
            input_hash=input_hash,
            output_hash=output_hash,
            operation_hash=operation_hash,
            state=state,
            path=Path(path),
            created_at=created_at,
        )

    @staticmethod
    def _preprocessing_model_record_from_row(row: Optional[tuple]) -> Optional[PreprocessingModelCacheIndexRecord]:
        if row is None:
            return None

        model_hash, operation_hash, input_hash, path, created_at, step_order, step_name, method, features_idx = row
        return PreprocessingModelCacheIndexRecord(
            model_hash=model_hash,
            operation_hash=operation_hash,
            input_hash=input_hash,
            path=Path(path),
            created_at=created_at,
            step_order=step_order,
            step_name=step_name,
            method=method,
            features_idx=CacheIndexDB._features_idx_from_json(features_idx),
        )
    
    @staticmethod
    def _preprocessing_plan_record_from_row(row: Optional[tuple]) -> Optional[PreprocessingPlanCacheIndexRecord]:
        if row is None:
            return None

        plan_hash, path, created_at = row
        return PreprocessingPlanCacheIndexRecord(
            plan_hash=plan_hash,
            path=Path(path),
            created_at=created_at,
        )

    def _ensure_tensor_table_schema(self, cur: sqlite3.Cursor) -> None:
        columns = self._table_columns(cur, self.TENSOR_DATA_TABLE)
        if "state" not in columns:
            cur.execute(
                f"ALTER TABLE {self.TENSOR_DATA_TABLE} "
                "ADD COLUMN state TEXT NOT NULL DEFAULT 'fit';"
            )

    def _ensure_preprocessing_models_table_schema(self, cur: sqlite3.Cursor) -> None:
        columns = self._table_columns(cur, self.PREPROCESSING_MODELS_TABLE)
        required_columns = {"step_order", "step_name", "method", "features_idx"}
        primary_key_columns = {
            name for name, info in columns.items()
            if info["pk"] > 0
        }

        if required_columns.issubset(columns) and "model_hash" in primary_key_columns:
            return

        cur.execute(f"DROP TABLE IF EXISTS {self.PREPROCESSING_MODELS_TABLE};")
        cur.execute(
            f"""
            CREATE TABLE {self.PREPROCESSING_MODELS_TABLE} (
                model_hash TEXT NOT NULL,
                operation_hash TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                path TEXT NOT NULL,
                step_order INTEGER NOT NULL DEFAULT 0,
                step_name TEXT,
                method TEXT,
                features_idx TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (input_hash, operation_hash, model_hash)
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
    def _table_columns(cur: sqlite3.Cursor, table_name: str) -> dict[str, dict[str, Any]]:
        cur.execute(f"PRAGMA table_info({table_name});")
        return {
            row[1]: {
                "type": row[2],
                "notnull": row[3],
                "default": row[4],
                "pk": row[5],
            }
            for row in cur.fetchall()
        }

    @staticmethod
    def _features_idx_to_json(features_idx: Optional[Any]) -> Optional[str]:
        if features_idx is None:
            return None
        return json.dumps(normalize_for_hash(features_idx), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _features_idx_from_json(features_idx: Optional[str]) -> Optional[Any]:
        if features_idx is None:
            return None
        return json.loads(features_idx)