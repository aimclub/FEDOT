from typing import Optional

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.caching.cacher import Cacher
from fedot.preprocessing.tools.index_mapping_tools import (update_index_mapping,
                                                           update_indices, create_index_mapping)
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.optional_planner import build_optional_plan
from fedot.preprocessing.tools.tools import update_handler_mapping, update_tensor_data
from fedot.core.caching.tracer import TraceBuilder, TraceStage
from fedot.core.caching.cache_loader import Loader


class OptionalService:
    """Parent service class for optional preprocessing pipelines.

    This class is a base (parent) implementation for specialized child services
    (OptionalTabularService, OptionalTSService). It defines shared logic for
    optional transformations that are configured by user strategy. The main difference
    from obligatory service is that optional service is requires ready TensorData as input.

    Processing sequence in `fit_transform`:
    1. Build optional preprocessing plan for the provided data and strategy.
    2. Initialize `PreparedData` from source tensor data.
    3. Resolve preprocessing handlers required by plan steps.
    4. For each step:
       - remap feature indices according to current index mapping;
       - fit and apply the step handler;
       - refresh index mapping after feature space changes.
    5. Return transformed `PreparedData`.
    """
    handler_mapping = {}
    plan: Optional[PreprocessingPlan] = None
    use_cache: bool = True

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def fit_transform(
        self,
        data: TensorData,
        optional_steps,
    ) -> TensorData:
        """Build and execute optional preprocessing plan.

        Args:
            data: Input tensor data with features, target and metadata.
            optional_steps: Mapping with optional preprocessing strategy by step
                type. Each step can be configured explicitly or auto-generated.

        Returns:
            TensorData updated after executing optional preprocessing steps.
        """

        self.plan = build_optional_plan(data, optional_steps)

        cacher = Cacher(use_cache=self.use_cache)
        cached_data = cacher.load_tensor_data(input_data=data, operation=self.plan)
        input_hash = cached_data.input_hash
        plan_hash = cached_data.operation_hash

        optional_idx_mapping = create_index_mapping(data.features)

        prepared_data = None

        if len(self.plan.steps) > 0:
            cacher.cache_preprocessing_plan(plan=self.plan, plan_hash=plan_hash)
            self.handler_mapping = update_handler_mapping(
                self.plan, self.handler_mapping)

            prepared_data = PreparedData(features=data.features,
                                         target=data.target,
                                         idx_mapping=optional_idx_mapping,
                                         ts_shape=data.ts_init_shape)
            for i, step in enumerate(self.plan.steps):
                actual_mapping = prepared_data.idx_mapping
                prepared_data.new_cols_dict = None
                step.features_idx = update_indices(
                    actual_mapping, step.features_idx)

                handler_cls = self.handler_mapping[step.step][step.method]
                handler = handler_cls(**step.step_args)
                prepared_data = handler.fit_transform(
                    prepared_data,
                    step.features_idx
                )

                prepared_data.idx_mapping = update_index_mapping(
                    actual_mapping,
                    step.features_idx,
                    prepared_data.features,
                    prepared_data.new_cols_dict
                )

                cacher.cache_preprocessing_model(
                    input_hash=input_hash,
                    model=handler,
                    operation_hash=plan_hash,
                    step_order=i,
                    step_name=step.step.value,
                    method=step.method.value if hasattr(step.method, "value") else str(step.method),
                    features_idx=step.features_idx,
                )
        
        result_tensor_data = update_tensor_data(data, prepared_data)
        
        responce = cacher.cache_tensor_data(
            output_data=result_tensor_data,
            input_hash=input_hash,
            operation_hash=plan_hash,
            state=result_tensor_data.state,
            trace_stage="optional_preprocessing",
        )
        result_tensor_data.fingerprint = responce.output_hash

        return result_tensor_data

    def transform(self, data) -> TensorData:
        trace_uuid = getattr(data, "trace_uuid", None)
        if trace_uuid is None:
            raise ValueError("trace_uuid is required for optional preprocessing in predict state.")

        cacher = Cacher(use_cache=self.use_cache)
        trace_builder = TraceBuilder.from_trace_uuid(trace_uuid)
        train_stage = self._get_train_optional_stage(trace_builder)
        self.plan = Loader.load(
            train_stage.operation_path,
            kind="preprocessing_plan",
        )

        optional_idx_mapping = create_index_mapping(data.features)
        prepared_data = PreparedData(features=data.features,
                                     target=data.target,
                                     idx_mapping=optional_idx_mapping,
                                     ts_shape=data.ts_init_shape)
        
        model_refs = sorted(train_stage.models, key=lambda model_ref: model_ref.step_order)
        for model_ref in model_refs:
            step = self.plan.steps[model_ref.step_order]
            actual_mapping = prepared_data.idx_mapping
            prepared_data.new_cols_dict = None

            handler = Loader.load(
                model_ref.model_path,
                kind="preprocessing_model",
            )
            prepared_data = handler.transform(prepared_data)

            prepared_data.idx_mapping = update_index_mapping(
                actual_mapping,
                step.features_idx,
                prepared_data.features,
                prepared_data.new_cols_dict,
            )

        result_tensor_data = update_tensor_data(data, prepared_data)
        responce = cacher.cache_tensor_data(
            output_data=result_tensor_data,
            input_hash=data.fingerprint,
            operation_hash=train_stage.operation_hash,
            state=result_tensor_data.state,
            trace_stage="optional_preprocessing",
        )
        result_tensor_data.fingerprint = responce.output_hash

        return result_tensor_data

    @staticmethod
    def _get_train_optional_stage(trace_builder: TraceBuilder) -> TraceStage:
        for stage in trace_builder.stages:
            if stage.stage == "optional_preprocessing":
                return stage
        raise ValueError(
            f"Trace {trace_builder.trace_id} does not contain obligatory preprocessing stage."
        )
