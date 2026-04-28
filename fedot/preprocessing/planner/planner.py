from dataclasses import dataclass, field
from typing import Optional, List

from fedot.preprocessing.tools.preprocessor_types import PreprocessingStep


@dataclass
class PreprocessingPlan:

    steps: List[PreprocessingStep] = field(default_factory=list)

    def add_step(self, step: Optional[PreprocessingStep] = None):
        if step is not None:
            if isinstance(step, List):
                self.steps.extend(step)
            else:
                self.steps.append(step)
