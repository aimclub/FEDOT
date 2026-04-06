from typing import Optional

from fedot.core.data.prepared_data import PreparedData

from fedot.preprocessing.planner import build_optional_plan, PreprocessingPlan
from fedot.preprocessing.mapping import PREPROCESSING_OPTIONAL_MAPPING

class OtionalPreprocessingService:

    plan: Optional[PreprocessingPlan] = None

    def fit_transform(self, data, pipline, optional_steps) -> PreparedData:
        self.plan = build_optional_plan(data, pipline, optional_steps)
        # steps = self.plan.steps.copy()
        prepared_data = None
        if len(self.plan.steps) > 0:
            prepared_data = PreparedData(features=data.features, target=data.target)
            for i, step in enumerate(self.plan.steps):
                handler_cls = PREPROCESSING_OPTIONAL_MAPPING[step.step][step.method]
                handler = handler_cls(**step.step_args)
                prepared_data = handler.fit_transform(
                    prepared_data,
                    step.features_idx
                )
                # TODO:caching handler
                # self.plan.steps[i]["model_hash"] = model_hash
        return prepared_data

    def transform(self, data, pipline, optional_steps, plan) -> PreparedData:
        for step in self.plan.steps:
            # TODO: get cached params
            # handler = PREPROCESSING_OPTIONAL_MAPPING[step.step][step.method](**params)
            # prepared = handler.transform(data)
            ...
        # return prepared
