import torch

from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.tools.index_mapping_tools import update_indices
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.prepared_data.prepared_data import PreparedData


def update_handler_mapping(plan: PreprocessingPlan,
                           handler_mapping: dict) -> dict:
    """Inject custom step implementations from plan into handler mapping.

    Args:
        plan: Preprocessing plan containing resolved steps.
        handler_mapping: Global mapping `{step_enum: {method_enum: handler_cls}}`.

    Returns:
        Updated handler mapping with `custom` step methods from current plan.
    """
    custom_dict = {}
    for step in plan.steps:
        if step.step == PreprocessingStepEnum.custom:
            custom_dict[step.method] = step.implementation

    if not custom_dict:
        return handler_mapping

    handler_mapping[PreprocessingStepEnum.custom] = custom_dict
    return handler_mapping


def get_used_idx_from_plan(plan: PreprocessingPlan) -> list:
    """Collect all non-target feature indices referenced by plan steps.

    Args:
        plan: Preprocessing plan to inspect.

    Returns:
        List of feature indices used by plan steps except target encoding.
    """
    used_idx = []

    if plan is None:
        return used_idx

    for step in plan.steps:
        if step.step == PreprocessingStepEnum.target_encoding:
            continue
        used_idx.extend(step.features_idx)
    return used_idx


def get_useful_idx(features_width: int,
                   plan: PreprocessingPlan,
                   categorical_idx: list,
                   idx_mapping: dict) -> tuple:
    """Get useful indices for the features.
    """
    idx = torch.arange(features_width, dtype=torch.int32)
    preprocessed_idx = get_used_idx_from_plan(plan)
    categorical_idx = update_indices(idx_mapping, categorical_idx)
    categorical_idx = list(set(categorical_idx) | set(preprocessed_idx))
    numerical_idx = list(set(range(features_width)) - set(categorical_idx))
    return idx, categorical_idx, numerical_idx


def update_tensor_data(data: TensorData, prepared_data: PreparedData) -> TensorData:
    """Update tensor data with prepared data.
    """
    data.features = prepared_data.features
    data.target = prepared_data.target
    data.idx_mapping = prepared_data.idx_mapping
    return data