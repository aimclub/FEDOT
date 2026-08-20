import sqlite3
from typing import Union

from golem.core.log import default_log
from golem.utilities.singleton_meta import SingletonMeta

from fedot.core.caching.operations_cache_db import OperationsCacheDB
from fedot.core.caching.preprocessing_cache_db import PreprocessingCacheDB
from fedot.core.caching.predictions_cache_db import PredictionsCacheDB

#  Sqlite failures caused by the environment instead of a bug in the caching code: a full disk, or a write lock
#  held by another process sharing the same cache file (sqlite reports SQLITE_BUSY as 'database is locked'
#  and SQLITE_LOCKED as 'database table is locked').
_EXPECTED_DB_ERROR_REASONS = ('disk is full', 'locked', 'busy')


def _is_expected_db_error(ex: BaseException) -> bool:
    return isinstance(ex, sqlite3.Error) and any(reason in str(ex) for reason in _EXPECTED_DB_ERROR_REASONS)


class BaseCache(metaclass=SingletonMeta):
    """
    Stores/loads data to increase performance.

    :param cache_db: specific DB for specific data
    """

    def __init__(self, cache_db: Union[OperationsCacheDB, PreprocessingCacheDB, PredictionsCacheDB]):
        self._db = cache_db
        self.log = default_log(__name__)

    @property
    def effectiveness_ratio(self):
        """
        Returns percent of how many elements were loaded instead of computing.
        """
        if self._db.use_stats:
            #  Result order corresponds to the order in self.db._effectiveness_keys
            eff_dct = {}
            try:
                returned_eff = self._db.get_effectiveness()
            except Exception as ex:
                self._handle_db_error(ex, 'Cache effectiveness can not be estimated')
                return eff_dct
            for key, hit, total in zip(self._db.get_effectiveness_keys()[::2], returned_eff[::2], returned_eff[1::2]):
                key = key.split('_')[0]
                eff_dct[key] = round(hit / total, 3) if total else 0.
            return eff_dct

    def _handle_db_error(self, ex: Exception, msg: str, level: str = 'warning'):
        """
        Reports a cache operation that could not be performed. Since the cache is an optimization, environment-caused
        sqlite failures are only logged and leave the caller with a cache miss; anything else is treated as a bug
        and is raised in a test session as usual.

        :param ex: exception the cache DB has failed with
        :param msg: description of the operation that was not performed
        :param level: logging level for the errors considered unexpected
        """
        if _is_expected_db_error(ex):
            self.log.warning(msg, exc_info=ex)
        else:
            self.log.log_or_raise(level, ValueError(msg))

    def reset(self, full_clean=False):
        """
        Drops all scores from working table and resets efficiency table values to zero.
        """
        self._db.reset(full_clean)

    def __len__(self):
        return len(self._db)
