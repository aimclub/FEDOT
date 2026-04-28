from typing import Optional

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.tools.index_mapping_tools import update_index_mapping, update_indices
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.planner.optional_planner import build_optional_plan
from fedot.preprocessing.tools.tools import update_handler_mapping


class OptionalService:
    handler_mapping = {}
    plan: Optional[PreprocessingPlan] = None

    def fit_transform(self, data: TensorData, optional_steps) -> PreparedData:
        self.plan = build_optional_plan(data, optional_steps)

        prepared_data = None

        if len(self.plan.steps) > 0:
            self.handler_mapping = update_handler_mapping(self.plan, self.handler_mapping)

            prepared_data = PreparedData(features=data.features, 
                                         target=data.target, 
                                         idx_mapping=data.idx_mapping, 
                                         ts_shape=data.ts_init_shape)
            for i, step in enumerate(self.plan.steps):
                actual_mapping = prepared_data.idx_mapping
                prepared_data.new_cols_dict = None
                step.features_idx = update_indices(actual_mapping, step.features_idx)

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
        for step in self.plan.steps:
            # TODO romankuklo: get cached params
            # handler = PREPROCESSING_OPTIONAL_MAPPING[step.step][step.method](**params)
            # prepared = handler.transform(data)
            ...
        # return prepared