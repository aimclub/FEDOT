from typing import Any, Optional, Union, List

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
from fedot.core.caching.tracer import TraceBuilder
from fedot.preprocessing.planner import PreprocessingPlan
from fedot.core.data.common.enums import StateEnum
from fedot.core.caching.enums import CacheModeEnum
from fedot.core.caching.normalization import normilize_cleaning_strategy
from fedot.core.caching.cache_cleaner import CacheCleaner


logger = logging.getLogger(__name__)


class Cacher:
    def __init__(
        self, 
        index_db: Optional[CacheIndexDB] = None, 
        use_cache: bool = True,
    ):
        self.index_db = index_db or CacheIndexDB()
        self.use_cache = use_cache

    def cache_tensor_data(
        self,
        output_data: Any,
        output_hash: str=None,
        input_data: Any=None,
        input_hash: str=None,
        operation: Any=None,
        operation_hash: str=None,
        state: Union[str, StateEnum] = "fit",
        trace_stage: str = None,
    ) -> TensorDataCacheIndexRecord:

        state=state.value if hasattr(state, "value") else str(state)

        if input_hash is None:
            input_hash = Hasher.hash(input_data)
        if operation_hash is None:
            operation_hash = Hasher.hash(operation)
        if output_hash is None:
            output_hash = Hasher.hash(output_data)
        
        if output_data.fingerprint != output_hash:
            output_data.fingerprint = output_hash

        trace_builder = None
        if trace_stage is not None and state == StateEnum.FIT.value:
            trace_builder = self._get_trace_builder(output_data, input_hash)
            output_data.trace_uuid = trace_builder.trace_id

        if self.use_cache:
            saver_response = Saver.save(output_data, output_hash)
            if not saver_response.success:
                logger.error(f"Failed to add index due to failed save TensorData: {input_hash} {operation_hash}")
                return None
            path = saver_response.path
        else:
            path = None


        result = self.index_db.add_tensor_data(
            input_hash=input_hash,
            output_hash=output_hash,
            operation_hash=operation_hash,
            path=path,
            state=state,
        )

        if trace_builder is not None:
            trace_builder.add_stage(
                stage=trace_stage,
                input_hash=input_hash,
                operation_hash=operation_hash,
            )
            trace_builder.save(final_output_hash=output_hash)

        return result

    def load_tensor_data(self, input_data: Any, operation: Any, target: Any = None) -> Optional[Any]:
        input_hash = Hasher.hash(input_data, target=target) if target is not None else Hasher.hash(input_data)
        operation_hash = Hasher.hash(operation)
        if not self.use_cache:
            return DataCacherLoaderResponse(
                data=None,
                input_hash=input_hash,
                operation_hash=operation_hash,
                success=False,
            )
        record = self.index_db.get_tensor_data(input_hash, operation_hash)

        if record is None or record.path is None or not record.path.exists():
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
    
    def clear_cache(
        self,
        mode: Union[CacheModeEnum, str] = CacheModeEnum.TENSOR_DATA,
        tensor_data_hashes: Optional[Union[str, List[str]]] = None,
        ratio_first_tensor_data: Optional[int] = None,
    ):
        norm_params = normilize_cleaning_strategy(
            mode=mode,
            tensor_data_hashes=tensor_data_hashes,
            ratio_first_tensor_data=ratio_first_tensor_data,
        )

        cleaner = CacheCleaner(self.index_db)
        if norm_params.mode == CacheModeEnum.ALL:
            cleaner.clear_all()
        elif norm_params.mode == CacheModeEnum.TENSOR_DATA:
            cleaner.clear_tensor_data(norm_params.tensor_data_hashes)
        else:
            cleaner.clear_first_n_tensor_data(norm_params.ratio_first_tensor_data)

    def _get_trace_builder(self, output_data: Any, raw_fingerprint: str) -> TraceBuilder:
        trace_uuid = getattr(output_data, "trace_uuid", None)
        if trace_uuid is not None:
            return TraceBuilder.from_trace_uuid(trace_uuid, index_db=self.index_db)

        return TraceBuilder(raw_fingerprint=raw_fingerprint, index_db=self.index_db)
