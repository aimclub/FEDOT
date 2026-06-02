from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.preprocessing.tools.index_mapping_tools import update_index_mapping, update_indices
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.obligatory_planner import build_obligatory_plan
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.tools.tools import update_handler_mapping
from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OBLIGATORY_MAPPING
from fedot.core.data.common.types import ArrayType
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.tracer import TraceBuilder, TraceStage

if TYPE_CHECKING:
    from fedot.core.data.tensor_data.tensor_data import TensorData


@dataclass
class ObligatoryPreprocessResult:
    prepared_data: Optional[PreparedData] = None
    tensor_data: Optional["TensorData"] = None
    plan_hash: Optional[str] = None
    raw_fingerprint: Optional[str] = None

    @classmethod
    def from_prepared_data(cls, prepared_data: PreparedData) -> "ObligatoryPreprocessResult":
        return cls(prepared_data=prepared_data)

    @classmethod
    def from_tensor_data(cls, tensor_data: "TensorData") -> "ObligatoryPreprocessResult":
        return cls(tensor_data=tensor_data)

    @property
    def has_tensor_data(self) -> bool:
        return self.tensor_data is not None


class ObligatoryService:
    """Parent service class for obligatory preprocessing pipelines.

    This class is a base (parent) implementation used by specialized child
    services (ObligatoryTabularService). It defines the common
    orchestration flow for mandatory preprocessing that must be applied before
    model training or inference.

    Processing sequence in `fit_transform`:
    1. Build obligatory preprocessing plan from input data and strategy params.
    2. Wrap raw features/target into `PreparedData` with index mapping.
    3. Resolve handler classes for all plan steps.
    4. For each step:
       - remap logical feature indices to current feature positions;
       - fit and apply handler transformation;
       - update feature index mapping after column changes.
    5. Return transformed `PreparedData`.
    """
    handler_mapping = {}
    plan: Optional[PreprocessingPlan] = None

    def fit_transform(self, features: ArrayType, target: ArrayType, params: dict) -> ObligatoryPreprocessResult:
        """Build and execute obligatory preprocessing plan.

        Args:
            features: Input feature matrix/tensor.
            target: Target values aligned with `features`.
            params: Preprocessing configuration dictionary with strategies and
                metadata (including feature names and index mapping).

        Returns:
            Result with either prepared data after preprocessing or ready
            TensorData restored from cache.
        """
        prepared_data = None

        self.plan = build_obligatory_plan(features, target, params)

        cacher = Cacher()
        cached_data = cacher.load_tensor_data(input_data=features, target=target, operation=self.plan)
        raw_fingerprint = cached_data.input_hash
        plan_hash = cached_data.operation_hash
        if cached_data.success:
            cached_data.data.features_names = params.get('features_names')
            return ObligatoryPreprocessResult(
                tensor_data=cached_data.data,
                plan_hash=plan_hash,
                raw_fingerprint=raw_fingerprint,
            )

        cacher.cache_preprocessing_plan(plan=self.plan, plan_hash=plan_hash)

        prepared_data = PreparedData(features=features,
                                     target=target,
                                     idx_mapping=params['idx_mapping'],
                                     ts_shape=features.shape)

        if len(self.plan.steps) > 0:
            self.handler_mapping = update_handler_mapping(
                self.plan, self.handler_mapping)

            for i, step in enumerate(self.plan.steps):
                actual_mapping = prepared_data.idx_mapping
                prepared_data.new_cols_dict = None

                handler_cls = self.handler_mapping[step.step][step.method]
                handler = handler_cls(**step.step_args)

                if step.step == PreprocessingStepEnum.target_encoding:
                    prepared_data_target = PreparedData(features=target)
                    prepared_data = handler.fit_transform(
                        prepared_data_target,
                        step.features_idx
                    )
                    prepared_data.target = prepared_data_target.features

                    cacher.cache_preprocessing_model(
                        input_hash=raw_fingerprint,
                        model=handler,
                        operation_hash=plan_hash,
                        step_order=i,
                        step_name=step.step.value,
                        method=step.method.value if hasattr(step.method, "value") else str(step.method),
                        features_idx=step.features_idx,
                    )
                    continue

                step.features_idx = update_indices(
                    actual_mapping, step.features_idx)

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
                    input_hash=raw_fingerprint,
                    model=handler,
                    operation_hash=plan_hash,
                    step_order=i,
                    step_name=step.step.value,
                    method=step.method.value if hasattr(step.method, "value") else str(step.method),
                    features_idx=step.features_idx,
                )

        return ObligatoryPreprocessResult(
            prepared_data=prepared_data,
            plan_hash=plan_hash,
            raw_fingerprint=raw_fingerprint,
        )


    def transform(
        self,
        features: ArrayType,
        target: ArrayType,
        idx_mapping: dict[int, int],
        trace_uuid: str,
    ) -> ObligatoryPreprocessResult:
        if trace_uuid is None:
            raise ValueError("trace_uuid is required for obligatory preprocessing in predict state.")

        trace_builder = TraceBuilder.from_trace_uuid(trace_uuid)
        train_stage = self._get_train_obligatory_stage(trace_builder)
        self.plan = Loader.load(
            train_stage.operation_path,
            kind="preprocessing_plan",
        )

        raw_fingerprint = Hasher.hash(features, target=target)
        prepared_data = PreparedData(
            features=features,
            target=target,
            idx_mapping=idx_mapping,
            ts_shape=features.shape,
        )

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

            if step.step == PreprocessingStepEnum.target_encoding:
                continue

            features_idx = model_ref.features_idx if model_ref.features_idx is not None else step.features_idx
            prepared_data.idx_mapping = update_index_mapping(
                actual_mapping,
                features_idx,
                prepared_data.features,
                prepared_data.new_cols_dict,
            )

        return ObligatoryPreprocessResult(
            prepared_data=prepared_data,
            plan_hash=train_stage.operation_hash,
            raw_fingerprint=raw_fingerprint,
        )

    @staticmethod
    def _get_train_obligatory_stage(trace_builder: TraceBuilder) -> TraceStage:
        for stage in trace_builder.stages:
            if stage.stage == "obligatory_preprocessing":
                return stage
        raise ValueError(
            f"Trace {trace_builder.trace_id} does not contain obligatory preprocessing stage."
        )