from typing import Optional

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.preprocessing_state import PreprocessingState

from fedot.preprocessing.planner import build_optional_plan, PreprocessingPlan
from fedot.preprocessing.mapping import PREPROCESSING_MAPPING

class BasePreprocessingService:

    plan: Optional[PreprocessingPlan] = None

    def fit(self, data, pipline, optional_steps) -> PreparedData:
        self.plan = build_optional_plan(data, pipline, optional_steps)
        steps = self.plan.steps.copy()
        for i, step in enumerate(steps):
            handler = PREPROCESSING_MAPPING[step.step][step.method]
            prepared_data, model_hash = handler.fit(data)
            self.plan.steps[i]["model_hash"] = model_hash

        return prepared_data

    def transform(self, data, pipline, optional_steps, plan) -> PreparedData:
        plan = build_optional_plan(data, pipline, optional_steps)
        for step in plan.steps:
            handler = PREPROCESSING_MAPPING[step.step][step.method]
            prepared = handler.transform(data, step.model_hash)
        return prepared
