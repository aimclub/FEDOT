
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

    def save_pipeline_prediction(self, pipeline: "Pipeline", outputData: OutputData, fold_id: int):
        """
        Save the prediction results of a pipeline to the cache.
        """
        type = "pipeline"
        uid = f"{type}_{self._create_uid(pipeline, fold_id)}"
        self._save_prediction(uid, type, outputData)

    def load_pipeline_prediction(self, pipeline: "Pipeline", fold_id: int):
        """
        Load the prediction results of a pipeline to the cache.
        """
        type = "pipeline"
        uid = f"{type}_{self._create_uid(pipeline, fold_id)}"
        return self._load_prediction(uid, type)

    def save_node_fit_prediction():
        """
        Save the prediction results of a fitted node to the cache.
        """
        pass

    def load_node_fit_prediction():
        """
        Save the prediction results of a fitted node to the cache.
        """
        pass

    def save_node_prediction(self, descriptive_id: str, output_mode: str, fold_id: int, outputData: OutputData,):
        """
        Save the prediction results of a node.
        """
        type = "pred"
        uid = f"{type}_{descriptive_id}_{output_mode}_{fold_id}"
        self._save_prediction(uid, type, outputData)

    def load_node_prediction(self, descriptive_id: str, output_mode: str, fold_id: int):
        """
        Load the prediction results of a node.
        """
        type = "pred"
        uid = f"{type}_{descriptive_id}_{output_mode}_{fold_id}"
        return self._load_prediction(uid, type)

    def _save_prediction(self, uid: str, type: str, outputData: OutputData):
        self.log.debug(f"--- SAVE prediction cache: {uid}")
        self._db.add_prediction(uid, type, outputData)

    def _load_prediction(self, uid: str, type: str):
        outputData = self._db.get_prediction(uid, type)
        self.log.debug(f"--- {'MISS' if outputData is None else 'HIT'} prediction cache: {uid}")
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
        from fedot.core.pipelines.pipeline import Pipeline
        if isinstance(pipeline, Pipeline):
            for node in pipeline.nodes:
                base_uid += f"{node.descriptive_id}_"
        else:
            base_uid += f"{pipeline.descriptive_id}_"
        if fold_id is not None:
            base_uid += f"{fold_id}"
        return base_uid
