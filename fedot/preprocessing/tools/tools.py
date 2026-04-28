from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


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
