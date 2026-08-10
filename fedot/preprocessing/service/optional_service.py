from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Sequence

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.caching.cacher import Cacher
from fedot.preprocessing.tools.index_mapping_tools import (update_index_mapping,
                                                           update_indices, create_index_mapping)
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.planner.optional_planner import build_optional_plan
from fedot.preprocessing.schemas import validate_optional_service_predict_ready
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.tools.tools import (
    copy_handler_mapping,
    update_handler_mapping,
    update_tensor_data,
)
from fedot.core.caching.cache_loader import Loader


class OptionalService:
    """Parent service class for optional preprocessing pipelines.

    This class is a base (parent) implementation for specialized child services
    (OptionalTabularService, OptionalTSService). It defines shared logic for
    optional transformations that are configured by user strategy. The main difference
    from obligatory service is that optional service is requires ready TensorData as input.

    Fitted-handler contract:
    - ``use_cache=False``: keep fitted handlers in ``fitted_handlers`` (in memory).
      The fitted service (including handlers) must be pickled with the pipeline node.
    - ``use_cache=True``: persist each fitted handler via ``Cacher`` and keep only
      disk path refs on the service; ``predict`` loads handlers through ``Loader``.
      Path refs travel with the pickled node; artifacts must still exist under cache.

    Processing sequence:
    1. Build optional preprocessing plan for the provided data and strategy.
    2. Initialize `PreparedData` from source tensor data.
    3. Resolve preprocessing handlers required by plan steps.
    4. During `fit`, train each handler in execution order and either store it in
       memory or cache it to disk (depending on ``use_cache``).
    5. During `predict`, resolve handlers (memory or cache) and apply them without
       refitting. For each step:
       - remap feature indices according to current index mapping;
       - apply the fitted step handler;
       - refresh index mapping after feature space changes.
    6. Return a new transformed `TensorData`.
    """
    handler_mapping = {}

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.plan = None
        self.fitted_handlers: Optional[List[Any]] = None
        self._cached_handler_paths: List[Path] = []
        self._input_hash = None
        self._plan_hash = None
        # Instance copy so fit/custom steps cannot mutate the class-/module-level mapping.
        self.handler_mapping = copy_handler_mapping(type(self).handler_mapping)

    def fit(
        self,
        data: TensorData,
        optional_steps: Optional[Sequence[PreprocessingStepEnum]] = None,
    ) -> 'OptionalService':
        if optional_steps is None:
            optional_steps = [PreprocessingStepEnum.imputation, PreprocessingStepEnum.scaling]

        self.plan = build_optional_plan(data, optional_steps)

        cacher = Cacher(use_cache=self.use_cache)
        cached_data = cacher.load_tensor_data(input_data=data, operation=self.plan)
        self._input_hash = cached_data.input_hash
        self._plan_hash = cached_data.operation_hash
        self.fitted_handlers = []
        self._cached_handler_paths = []

        cacher.cache_preprocessing_plan(plan=self.plan, plan_hash=self._plan_hash)
        self.handler_mapping = update_handler_mapping(self.plan, self.handler_mapping)
        prepared_data = self._create_prepared_data(deepcopy(data))

        for i, step in enumerate(self.plan.steps):
            actual_mapping = prepared_data.idx_mapping
            prepared_data.new_cols_dict = None
            step.features_idx = update_indices(actual_mapping, step.features_idx)

            handler_cls = self.handler_mapping[step.step][step.method]
            handler = handler_cls(**step.step_args)
            prepared_data = handler.fit_transform(prepared_data, step.features_idx)

            prepared_data.idx_mapping = update_index_mapping(
                actual_mapping,
                step.features_idx,
                prepared_data.features,
                prepared_data.new_cols_dict,
            )

            # Always register the fitted handler in the cache index.
            # Disk persistence is controlled by Cacher.use_cache (path may be None).
            record = cacher.cache_preprocessing_model(
                input_hash=self._input_hash,
                model=handler,
                operation_hash=self._plan_hash,
                step_order=i,
                step_name=step.step.value,
                method=step.method.value if hasattr(step.method, "value") else str(step.method),
                features_idx=step.features_idx,
            )
            if record.path is not None:
                self._cached_handler_paths.append(Path(record.path))
            else:
                self.fitted_handlers.append(handler)

        if self._cached_handler_paths:
            # Handlers live on disk; keep only path refs on the service instance.
            self.fitted_handlers = None

        return self

    def predict(self, data: TensorData) -> TensorData:
        validate_optional_service_predict_ready(
            plan=self.plan,
            fitted_handlers=self.fitted_handlers,
            cached_handler_paths=self._cached_handler_paths,
        )

        init_fingerprint = data.fingerprint
        prepared_data = self._create_prepared_data(data)
        handlers = self._resolve_handlers()

        for step, handler in zip(self.plan.steps, handlers, strict=True):
            actual_mapping = prepared_data.idx_mapping
            prepared_data.new_cols_dict = None
            prepared_data = handler.transform(prepared_data)

            prepared_data.idx_mapping = update_index_mapping(
                actual_mapping,
                step.features_idx,
                prepared_data.features,
                prepared_data.new_cols_dict,
            )

        data = update_tensor_data(data, prepared_data)
        cacher = Cacher(use_cache=self.use_cache)
        response = cacher.cache_tensor_data(
            output_data=data,
            input_hash=init_fingerprint,
            operation_hash=self._plan_hash,
            state=data.state,
            trace_stage="optional_preprocessing",
        )
        data.fingerprint = response.output_hash

        return data

    def _resolve_handlers(self) -> Sequence[Any]:
        if self.fitted_handlers is not None:
            return self.fitted_handlers
        if not self._cached_handler_paths:
            raise RuntimeError(
                'Optional preprocessing handlers are missing both in memory and cache refs.'
            )
        return [
            Loader.load(str(path), kind='preprocessing_model')
            for path in self._cached_handler_paths
        ]

    @staticmethod
    def _create_prepared_data(data: TensorData) -> PreparedData:
        return PreparedData(
            features=data.features,
            target=data.target,
            idx_mapping=create_index_mapping(data.features),
            ts_shape=data.ts_init_shape,
        )
