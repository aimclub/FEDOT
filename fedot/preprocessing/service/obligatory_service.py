from typing import Optional

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.tools.index_mapping_tools import update_index_mapping, update_indices
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.obligatory_planner import build_obligatory_plan
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.tools.tools import update_handler_mapping
from fedot.preprocessing.tools.methods_mapping import PREPROCESSING_OBLIGATORY_MAPPING
from fedot.core.data.complex_types import ArrayType


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

    def fit_transform(self, features: ArrayType, target: ArrayType, params: dict) -> PreparedData:
        """Build and execute obligatory preprocessing plan.

        Args:
            features: Input feature matrix/tensor.
            target: Target values aligned with `features`.
            params: Preprocessing configuration dictionary with strategies and
                metadata (including feature names and index mapping).

        Returns:
            Prepared data after all obligatory preprocessing steps.
        """
        prepared_data = None

        self.plan = build_obligatory_plan(features, target, params)

        prepared_data = PreparedData(features=features, 
                                     target=target, 
                                     idx_mapping=params['idx_mapping'], 
                                     ts_shape=features.shape)

        if len(self.plan.steps) > 0:
            self.handler_mapping = update_handler_mapping(self.plan, self.handler_mapping)

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
                    continue

                step.features_idx = update_indices(actual_mapping, step.features_idx)

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
                # TODO romankuklo: caching handler
                # self.plan.steps[i]["model_hash"] = model_hash
        return prepared_data