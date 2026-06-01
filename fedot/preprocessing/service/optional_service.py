from typing import Optional

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.data.common.enums import StateEnum
from fedot.preprocessing.tools.index_mapping_tools import (update_index_mapping,
                                                           update_indices, create_index_mapping)
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.optional_planner import build_optional_plan
from fedot.preprocessing.tools.tools import update_handler_mapping


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

    def fit_transform(
        self,
        data: TensorData,
        optional_steps,
        trace_builder: Optional[TraceBuilder] = None,
    ) -> PreparedData:
        """Build and execute optional preprocessing plan.

        Args:
            data: Input tensor data with features, target and metadata.
            optional_steps: Mapping with optional preprocessing strategy by step
                type. Each step can be configured explicitly or auto-generated.

        Returns:
            Prepared data after executing optional preprocessing steps.
        """
        self.plan = build_optional_plan(data, optional_steps)
        trace_builder = trace_builder or getattr(data, "trace_builder", None)
        cacher = Cacher()
        input_hash = data.ready_fingerprint or Hasher.hash(data)
        operation_hash = Hasher.hash(self.plan)

        optional_idx_mapping = create_index_mapping(data.features)

        prepared_data = None

        if len(self.plan.steps) > 0:
            cacher.cache_preprocessing_plan(plan=self.plan, plan_hash=operation_hash)
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
                # TODO:caching handler
                # self.plan.steps[i]["model_hash"] = model_hash
        return prepared_data

    def transform(self, data, optional_steps, plan) -> PreparedData:
        """Placeholder for transform-only execution with cached plan/handlers.

        Args:
            data: Input data to transform.
            optional_steps: Optional strategy configuration.
            plan: Pre-built preprocessing plan.

        Returns:
            Transformed prepared data when implementation is completed.
        """
        for step in self.plan.steps:
            # TODO romankuklo: get cached params
            # handler = PREPROCESSING_OPTIONAL_MAPPING[step.step][step.method](**params)
            # prepared = handler.transform(data)
            ...
        # return prepared
