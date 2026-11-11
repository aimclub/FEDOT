from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TensorMergeInputPlan:
    """Select the semantic output field of a TensorData parent node."""

    use_predict: bool


def plan_tensor_merge_input(
    has_prediction: bool,
    parent_is_model: Optional[bool] = None,
) -> TensorMergeInputPlan:
    """Select a parent result by operation role, with a direct-call fallback."""
    if parent_is_model is True and not has_prediction:
        raise ValueError('TensorData model output must provide predict')
    use_predict = has_prediction if parent_is_model is None else parent_is_model
    return TensorMergeInputPlan(use_predict=use_predict)
