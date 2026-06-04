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
    """
    Facade for disk-backed preprocessing and tensor-data cache.

    `Cacher` coordinates hashing, artifact persistence, SQLite indexing, and
    optional trace manifests. Callers interact with a single object instead of
    wiring `Hasher`, `Saver`, `Loader`, `CacheIndexDB`, and `TraceBuilder`
    directly.
    """

    def __init__(
        self,
        index_db: Optional[CacheIndexDB] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the cache facade.

        Args:
            index_db: SQLite index instance. When omitted, a default index under
                `CACHE_DIR` is created.
            use_cache: When ``False``, index rows may be written without saving
                tensor artifacts to disk (trace-only mode).
        """
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
        """
        Persist ``TensorData`` and register it in the cache index.

        Hashes are computed when omitted. During ``fit`` with ``trace_stage``,
        a trace manifest stage is appended and ``output_data.trace_uuid`` is set.

        Args:
            output_data: Prepared tensor container to cache.
            output_hash: Fingerprint of ``output_data``. Computed when omitted.
            input_data: Raw input used to derive ``input_hash`` when omitted.
            input_hash: Fingerprint of the preprocessing input.
            operation: Preprocessing plan or operation object.
            operation_hash: Fingerprint of ``operation``.
            state: Pipeline state label (for example ``fit`` or ``predict``).
            trace_stage: Trace stage name; used only for ``fit`` state.

        Returns:
            Saved index record, or ``None`` if the tensor file could not be written.
        """
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
        """
        Load cached ``TensorData`` for an input/operation pair.

        Args:
            input_data: Raw or prepared input used to compute ``input_hash``.
            operation: Preprocessing plan or operation object.
            target: Optional target array included in the input fingerprint.

        Returns:
            ``DataCacherLoaderResponse`` with ``success=False`` when cache is
            disabled, the index row is missing, or the artifact file is absent.
        """
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
        """
        Save a fitted preprocessing model and index it by content hashes.

        Reuses an existing on-disk model file when the same ``model_hash`` is
        already indexed and the path still exists.

        Args:
            input_data: Input used to derive ``input_hash`` when omitted.
            input_hash: Fingerprint of preprocessing input at this stage.
            model: Fitted preprocessing handler instance.
            model_hash: Fingerprint of ``model``. Computed when omitted.
            operation_hash: Fingerprint of the preprocessing plan.
            operation: Plan object used to derive ``operation_hash`` when omitted.
            step_order: Order of the model within the preprocessing stage.
            step_name: Optional human-readable step label for tracing.
            method: Optional preprocessing method identifier.
            features_idx: Optional feature index metadata stored in the index.

        Returns:
            Index record pointing to the cached ``.pkl`` artifact.
        """
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
        """
        Load a cached preprocessing model for an input/operation pair.

        Args:
            input_data: Input used to derive ``input_hash`` when omitted.
            input_hash: Fingerprint of preprocessing input.
            operation: Plan object used to derive ``operation_hash`` when omitted.
            operation_hash: Fingerprint of the preprocessing plan.

        Returns:
            Loaded model on success, otherwise ``DataCacherLoaderResponse`` with
            ``success=False``.
        """
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
        """
        Save a preprocessing plan and register it in the cache index.

        Args:
            plan: Preprocessing plan instance.
            plan_hash: Fingerprint of ``plan``. Computed when omitted.

        Returns:
            Existing or newly created index record for the plan artifact.
        """
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
        """
        Remove cached artifacts according to the selected cleaning mode.

        Args:
            mode: Cleaning strategy (`all`, `tensor_data`, `first_n_tensor_data`).
            tensor_data_hashes: Output hashes to remove for ``tensor_data`` mode.
                When omitted, all ``.pt`` files under ``tensor_data/`` are used.
            ratio_first_tensor_data: Number or fraction of oldest tensor artifacts
                to remove for ``first_n_tensor_data`` mode.
        """
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
        """
        Return a trace builder for continuing or starting a preprocessing trace.

        Args:
            output_data: Tensor container that may already carry ``trace_uuid``.
            raw_fingerprint: Fingerprint of the raw input at the first stage.

        Returns:
            Existing trace loaded by UUID, or a new ``TraceBuilder`` instance.
        """
        trace_uuid = getattr(output_data, "trace_uuid", None)
        if trace_uuid is not None:
            return TraceBuilder.from_trace_uuid(trace_uuid, index_db=self.index_db)

        return TraceBuilder(raw_fingerprint=raw_fingerprint, index_db=self.index_db)
