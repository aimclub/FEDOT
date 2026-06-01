from typing import Any, Optional

import logging
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.index_db import (
    CacheIndexDB,
    PreprocessingModelCacheIndexRecord,
    TensorDataCacheIndexRecord,
    PreprocessingPlanCacheIndexRecord,
)
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.cache_saver import Saver
from fedot.core.caching.responses import DataCacherLoaderResponse
from fedot.preprocessing.planner import PreprocessingPlan


logger = logging.getLogger(__name__)

class Cacher:
    def __init__(self, index_db: Optional[CacheIndexDB] = None):
        self.index_db = index_db or CacheIndexDB()

    def cache_tensor_data(
        self,
        output_data: Any,
        output_hash: str=None,
        input_data: Any=None,
        input_hash: str=None,
        operation: Any=None,
        operation_hash: str=None,
        state: str = "fit",
    ) -> TensorDataCacheIndexRecord:
        if input_hash is None:
            input_hash = Hasher.hash(input_data)
        if operation_hash is None:
            operation_hash = Hasher.hash(operation)
        if output_hash is None:
            output_hash = Hasher.hash(output_data)

        saver_response = Saver.save(output_data, output_hash)

        if not saver_response.success:
            logger.error(f"Failed to add index due to failed save TensorData: {input_hash} {operation_hash}")
            return None

        result = self.index_db.add_tensor_data(
            input_hash=input_hash,
            output_hash=output_hash,
            operation_hash=operation_hash,
            path=saver_response.path,
            state=state,
        )

        return result

    def load_tensor_data(self, input_data: Any, target: Any, operation: Any) -> Optional[Any]:
        input_hash = Hasher.hash(input_data, target=target)
        operation_hash = Hasher.hash(operation)
        record = self.index_db.get_tensor_data(input_hash, operation_hash)

        if record is None or not record.path.exists():
            return DataCacherLoaderResponse(
                data=None,
                input_hash=input_hash,
                output_hash=None if record is None else record.output_hash,
                operation_hash=operation_hash,
                path=None if record is None else record.path,
                success=False,
            )

        loaded_data = Loader.load(str(record.path), record.output_hash, "tensor_data")
        success = False if loaded_data is None else True
        return DataCacherLoaderResponse(
            data=loaded_data,
            input_hash=input_hash,
            output_hash=record.output_hash,
            operation_hash=operation_hash,
            path=record.path,
            success=success,
        )

    def cache_preprocessing_model(
        self,
        input_data: Any=None,
        input_hash: str=None,
        model: Any=None,
        model_hash: str=None,
        operation_hash: str=None,
        operation: Any=None,
        step_order: int = 0,
        step_name: str = None,
        method: str = None,
        features_idx: Any = None,
    ) -> PreprocessingModelCacheIndexRecord:
        if input_hash is None:
            input_hash = Hasher.hash(input_data)
        if operation_hash is None:
            operation_hash = Hasher.hash(operation)
        if model_hash is None:
            model_hash = Hasher.hash(model)

        cached_record = self.index_db.get_preprocessing_model_by_model_hash(model_hash)
        if cached_record is not None and cached_record.path.exists():
            model_path = cached_record.path
        else:
            response = Saver.save(model, model_hash)
            model_path = response.path
        return self.index_db.add_preprocessing_model(
            model_hash=model_hash,
            operation_hash=operation_hash,
            input_hash=input_hash,
            path=model_path,
            step_order=step_order,
            step_name=step_name,
            method=method,
            features_idx=features_idx,
        )

    def load_preprocessing_model(self, 
        input_data: Any=None, 
        input_hash: str=None,
        operation: Any=None, 
        operation_hash: str=None,
    ) -> Optional[Any]:
        if input_hash is None:
            input_hash = Hasher.hash(input_data)
        if operation_hash is None:
            operation_hash = Hasher.hash(operation)

        record = self.index_db.get_preprocessing_model(input_hash, operation_hash)

        if record is None or not record.path.exists():
            return DataCacherLoaderResponse(
                model=None,
                input_hash=input_hash,
                model_hash=None if record is None else record.model_hash,
                operation_hash=operation_hash,
                path=None if record is None else record.path,
                success=False,
            )

        return Loader.load(str(record.path), record.model_hash, "preprocessing_model")


    def cache_preprocessing_plan(
        self,
        plan: PreprocessingPlan,
        plan_hash: str=None,
    ) -> PreprocessingPlanCacheIndexRecord:
        if plan_hash is None:
            plan_hash = Hasher.hash(plan)

        cached_record = self.index_db.get_preprocessing_plan(plan_hash=plan_hash)
        if cached_record is not None and cached_record.path.exists():
            return cached_record

        response = Saver.save(plan, plan_hash)
        return self.index_db.add_preprocessing_plan(
            plan_hash=plan_hash,
            path=response.path)