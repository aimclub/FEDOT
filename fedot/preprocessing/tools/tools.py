from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum


def update_handler_mapping(plan: PreprocessingPlan, 
                           handler_mapping: dict) -> dict:
    custom_dict = {}
    for step in plan.steps:
        if step.step == PreprocessingStepEnum.custom:
            custom_dict[step.method] = step.implementation
    
    if not custom_dict:
        return handler_mapping

    handler_mapping[PreprocessingStepEnum.custom] = custom_dict
    return handler_mapping


def get_used_idx_from_plan(plan: PreprocessingPlan) -> list:
    used_idx = []

    if plan is None:
        return used_idx

    for step in plan.steps:
        if step.step == PreprocessingStepEnum.target_encoding:
            continue
        used_idx.extend(step.features_idx)
    return used_idx
