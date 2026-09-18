import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.utils import CACHE_DIR


logger = logging.getLogger(__name__)


class CacheCleaner:
    """
    Remove cached artifacts from disk and keep the SQLite index consistent.

    File deletion is coordinated with ``CacheIndexDB`` and trace manifests are
    updated when tensor-data entries are removed.
    """

    def __init__(self, index_db: Optional[CacheIndexDB] = None):
        """
        Args:
            index_db: SQLite index used to resolve artifact paths. A default index
                is created when omitted.
        """
        self.index_db = index_db or CacheIndexDB()

    def clear_all(self) -> bool:
        """
        Delete the entire cache directory tree and reset index tables when present.

        Returns:
            ``True`` when the operation completes.
        """
        ensure_cache_dirs()
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        ensure_cache_dirs()
        if self.index_db.db_path.exists():
            self.index_db.clear_all_records()
        logger.info("Cache cleared")
        return True

    def clear_tensor_data(self, td_hashes: List[str]) -> bool:
        """
        Remove tensor-data artifacts and index rows for the given output hashes.

        Orphan ``.pt`` files without index rows are removed when present. Trace
        manifests are updated via ``TraceBuilder.update_according_to_cache``.

        Args:
            td_hashes: ``output_hash`` values identifying cached tensor artifacts.

        Returns:
            ``True`` when the operation completes.
        """
        ensure_cache_dirs()
        for output_hash in td_hashes:
            record = self.index_db.get_tensor_data_by_output_hash(output_hash)
            if record is None:
                fallback_path = CACHE_DIR / "tensor_data" / f"{output_hash}.pt"
                if fallback_path.is_file():
                    fallback_path.unlink()
                continue

            if record.path is not None and record.path.is_file():
                record.path.unlink()

            self.index_db.delete_tensor_data_by_output_hash(output_hash)
            logger.debug("TensorData cache entry removed: %s", output_hash)

        TraceBuilder.update_according_to_cache(td_hashes)
        return True

    def clear_first_n_tensor_data(self, n_first: int) -> bool:
        """
        Remove the oldest ``n_first`` tensor-data artifacts by file modification time.

        Args:
            n_first: Maximum number of ``.pt`` files to remove. Non-positive values
                are treated as a no-op.

        Returns:
            ``True`` when the operation completes.
        """
        ensure_cache_dirs()
        if n_first <= 0:
            return True
        return self.clear_tensor_data(self._oldest_tensor_data_hashes(n_first))

    @staticmethod
    def _oldest_tensor_data_hashes(n: int) -> List[str]:
        """
        List output hashes for the ``n`` oldest ``.pt`` files in ``tensor_data/``.

        Args:
            n: Maximum number of hashes to return.

        Returns:
            File stems ordered from oldest to newest.
        """
        tensor_dir = CACHE_DIR / "tensor_data"
        if not tensor_dir.exists():
            return []

        files = sorted(
            tensor_dir.glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        return [path.stem for path in files[:n]]
