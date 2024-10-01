import sqlite3
from typing import TYPE_CHECKING, List, Optional, Union

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

    def save_prediction(self, prediction: np.ndarray, uid: str):
        """
        Save the prediction for a given UID.

        :param prediction (np.ndarray): The prediction to be saved.
        :param uid (str): The unique identifier for the prediction.
        """
        try:
            self._db.add_prediction([(uid, prediction)])
        except Exception as ex:
            unexpected_exc = not (
                isinstance(ex, sqlite3.DatabaseError) and "disk is full" in str(ex)
            )
            self.log.warning(
                f"Predictions can not be saved: {ex}. Continue",
                exc=ex,
                raise_if_test=unexpected_exc,
            )

    def load_prediction(self, uid: str) -> np.ndarray:
        """
        Load the prediction data for the given unique identifier.
        :param uid (str): The unique identifier of the prediction data.
        :return np.ndarray: The loaded prediction data.
        """
        predict = self._db.get_prediction(uid)
        # TODO: restore OutputData from predict
        return predict

    def save_data(
        self,
        pipeline: "Pipeline",
        outputData: OutputData,
        fold_id: Optional[int] = None,
    ):
        """
        Save the pipeline data to the cache.

        :param pipeline: The pipeline data to be cached.
        :type pipeline: Pipeline
        :param outputData: The output data to be saved.
        :type outputData: OutputData
        :param fold_id: Optional part of the cache item UID (can be used to specify the number of CV fold).
        :type fold_id: Optional[int]
        """
        uid = self._create_uid(pipeline, fold_id)
        # TODO: save OutputData as a whole to the cache
        self.save_prediction(outputData.predict, uid)

    def try_load_data(
        self, pipeline: "Pipeline", fold_id: Optional[int] = None
    ) -> OutputData:
        # create parameter dosctring
        """
        Try to load data for the given pipeline and fold ID.

        :param  pipeline (Pipeline): The pipeline for which to load the data.
        :param fold_id (Optional[int]): The fold ID for which to load the data. Defaults to None.
        :return OutputData: The loaded data.
        """
        # TODO: implement loading of pipeline data
        uid = self._create_uid(pipeline, fold_id)
        self.load_prediction(uid)

    def _create_uid(
        self,
        pipeline: "Pipeline",
        fold_id: Optional[int] = None,
    ) -> str:
        """
        Generate a unique identifier for a pipeline.

        :param pipeline (Pipeline): The pipeline for which the unique identifier is generated.
        :param fold_id (Optional[int]): The fold ID (default: None).
        :return str: The unique identifier generated for the pipeline.
        """
        base_uid = ""
        for node in pipeline.nodes:
            base_uid += f"{node.descriptive_id}_"
        if fold_id is not None:
            base_uid += f"{fold_id}"
        return base_uid
