from typing import Union

from golem.core.log import default_log
from golem.utilities.singleton_meta import SingletonMeta

from fedot.core.caching.operations_cache_db import OperationsCacheDB
from fedot.core.caching.preprocessing_cache_db import PreprocessingCacheDB
from fedot.core.caching.predictions_cache_db import PredictionsCacheDB


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
            returned_eff = self._db.get_effectiveness()
            for key, hit, total in zip(self._db.get_effectiveness_keys()[::2], returned_eff[::2], returned_eff[1::2]):
                key = key.split('_')[0]
                eff_dct[key] = round(hit / total, 3) if total else 0.
            return eff_dct

    def reset(self, full_clean=False):
        """
        Drops all scores from working table and resets efficiency table values to zero.
        """
        self._db.reset(full_clean)

    def __len__(self):
        return len(self._db)
