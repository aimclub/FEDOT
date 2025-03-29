from typing import Optional

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.predictions_cache_db import PredictionsCacheDB
from fedot.core.data.data import OutputData


class PredictionsCache(BaseCache):
    """
    Stores/loads predictions to increase performance of calculations.

    :param cache_dir: path to the place where cache files should be stored.
    """

    FIT_TYPE = "fit"
    PRED_TYPE = "pred"

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None):
        super().__init__(PredictionsCacheDB(cache_dir, custom_pid))

    def save_node_prediction(
            self, descriptive_id: str, output_mode: str, fold_id: int, outputData: OutputData, is_fit: bool = False):
        """
        Save the prediction results of a node.
        """
        if "ransac" in descriptive_id:
            return
        type = self.FIT_TYPE if is_fit else self.PRED_TYPE
        uid = f"{type}_{descriptive_id}_{output_mode}_{fold_id}"
        self._save_prediction(uid, type, outputData)

    def load_node_prediction(self, descriptive_id: str, output_mode: str, fold_id: int, is_fit: bool = False):
        """
        Load the prediction results of a node.
        """
        if "ransac" in descriptive_id:
            return
        type = self.FIT_TYPE if is_fit else self.PRED_TYPE
        uid = f"{type}_{descriptive_id}_{output_mode}_{fold_id}"
        return self._load_prediction(uid, type)

    def _save_prediction(self, uid: str, type: str, outputData: OutputData):
        self.log.debug(f"--- SAVE prediction cache: {uid}")
        self._db.add_prediction(uid, type, outputData)

    def _load_prediction(self, uid: str, type: str):
        outputData = self._db.get_prediction(uid, type)
        self.log.debug(f"--- {'MISS' if outputData is None else 'HIT'} prediction cache: {uid}")
        return outputData
