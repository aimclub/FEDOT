from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from fedot.preprocessing.preprocessor_types import PreprocessingStep, PreprocessingStepEnum, ImputationMethodEnum
from fedot.core.data.tensordata import TensorData

from fedot.preprocessing.optional_steps import get_optional_steps


@dataclass
class PreprocessingPlan:

    steps: List[PreprocessingStep] = field(default_factory=list)

    def add_step(self, step: Optional[PreprocessingStep] = None):
        if step is not None:
            if isinstance(step, List):
                self.steps.extend(step)
            else:
                self.steps.append(step)


def build_optional_plan(data: TensorData, pipeline=None, optional_steps=None) -> PreprocessingPlan:

    optional_plan = PreprocessingPlan()

    for step_name in optional_steps.keys():
        step = get_optional_steps(step_name, data, pipeline, optional_steps[step_name])
        optional_plan.add_step(step)
    return optional_plan

