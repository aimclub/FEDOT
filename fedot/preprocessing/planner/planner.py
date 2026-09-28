from dataclasses import dataclass, field
from typing import Optional, List

from fedot.preprocessing.tools.preprocessor_types import PreprocessingStep


@dataclass
class PreprocessingPlan:

    """PreprocessingPlan definition used in preprocessing flow."""
    steps: List[PreprocessingStep] = field(default_factory=list)

    def add_step(self, step: Optional[PreprocessingStep] = None):
        """Append one or many preprocessing steps to plan.

        Args:
            step: Single preprocessing step, list of steps, or `None`.

        Returns:
            `None`. The method updates `self.steps` in place.
        """
        if step is not None:
            if isinstance(step, List):
                self.steps.extend(step)
            else:
                self.steps.append(step)
