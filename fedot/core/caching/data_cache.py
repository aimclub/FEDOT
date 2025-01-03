import sqlite3
from typing import TYPE_CHECKING, List, Optional, Union
from multiprocessing import Manager

import numpy as np

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.data_cache_db import DataCacheDB
from fedot.core.data.data import OutputData

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class DataCache(BaseCache):
    """
    Stores/loads predictions to increase performance of calculations.

    :param cache_dir: path to the place where cache files should be stored.
    """

    def __init__(self, cache_dir: Optional[str] = None, custom_pid=None):
        super().__init__(DataCacheDB(cache_dir, custom_pid))
        self.manager = Manager()
        self.metrics_cache = self.manager.dict()
        if self._db.use_stats:
            self.metrics_cache["metrics_hit"] = 0
            self.metrics_cache["metrics_total"] = 0
            self.metrics_stats = {}

    def save_metric(self, pipeline, metric, fold_id=None):
        uid = self._create_uid(pipeline, fold_id)
        self.metrics_cache[uid] = metric
        if self._db.use_stats:
            self.metrics_stats[uid] = 0

    def load_metric(self, pipeline, fold_id=None):
        uid = self._create_uid(pipeline, fold_id)
        result = self.metrics_cache.get(uid, None)
        if self._db.use_stats:
            if result:
                self.metrics_cache["metrics_hit"] += 1
                self.metrics_stats[uid] += 1
            self.metrics_cache["metrics_total"] += 1
        return result

    @property
    def get_effectiveness_metrics_ratio(self):
        if self._db.use_stats:
            eff_dct = {}
            hit, total = self.metrics_cache["metrics_hit"], self.metrics_cache["metrics_total"]
            eff_dct["metrics"] = round(hit / total, 3) if total else 0
            return eff_dct

    def save_predicted(self, pipeline: "Pipeline", outputData: OutputData, fold_id: Optional[int] = None):
        """
        Save the prediction for a given UID.

        :param prediction (np.ndarray): The prediction to be saved.
        :param uid (str): The unique identifier for the prediction.
        """
        uid = self._create_uid(pipeline, fold_id)
        try:
            self._db.add_prediction(uid, outputData)
        except Exception as ex:
            unexpected_exc = not (
                isinstance(
                    ex, sqlite3.DatabaseError) and "disk is full" in str(ex)
            )
            self.log.warning(
                f"Predictions can not be saved: {ex}. Continue",
                exc=ex,
                raise_if_test=unexpected_exc,
            )

    def load_predicted(self, pipeline: "Pipeline", fold_id: Optional[int] = None) -> np.ndarray:
        """
        Load the prediction data for the given unique identifier.
        :param uid (str): The unique identifier of the prediction data.
        :return np.ndarray: The loaded prediction data.
        """
        uid = self._create_uid(pipeline, fold_id)
        outputData = self._db.get_prediction(uid)
        return outputData

    def _create_uid(
        self,
        pipeline: "Pipeline",  # TODO: pipeline or node
        fold_id: Optional[int] = None,
    ) -> str:
        """
        Generate a unique identifier for a pipeline.

        :param pipeline (Pipeline): The pipeline for which the unique identifier is generated.
        :param fold_id (Optional[int]): The fold ID (default: None).
        :return str: The unique identifier generated for the pipeline.
        """
        base_uid = ""
        # TODO: pipeline or node
        from fedot.core.pipelines.pipeline import Pipeline
        if isinstance(pipeline, Pipeline):
            for node in pipeline.nodes:
                base_uid += f"{node.descriptive_id}_"
        else:
            base_uid += f"{pipeline.description()}_"
        if fold_id is not None:
            base_uid += f"{fold_id}"
        return base_uid
