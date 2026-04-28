import torch
import logging

from fedot.preprocessing.planner.auto_create_step import AUTO_CREATE_STEP_MAPPING
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStep, PreprocessingStepEnum
from fedot.preprocessing.planner.planner import PreprocessingPlan


logger = logging.getLogger(__name__)


def get_steps_from_params(step_name: PreprocessingStepEnum, params):
    steps = []
    for step_params in params:
        step = PreprocessingStep(step_name, step_params['method'], step_params['features_idx'])
        if step_params['step_args'] is not None:
            step.step_args = step_params['step_args']
        
        implementation = step_params.get('implementation')
        if implementation is not None:
            step.implementation = step_params['implementation']
        steps.append(step)
    return steps


def is_imputation_needed(features: torch.Tensor) -> bool:
    """
    Check whether a torch.Tensor contains any NaN values.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    bool
        True if tensor contains at least one NaN, False otherwise.
    """
    return torch.isnan(features).any().item()


def get_imputation_step(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    if is_imputation_needed(data.features):
        if params is None:
            logger.info(f'Getting default params for step {step_name}')
            return AUTO_CREATE_STEP_MAPPING[step_name](data)
        else:
            steps = get_steps_from_params(step_name, params)
            return steps
    else:
        return None


def get_scaling_step(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    if len(data.numerical_idx) == 0:
        logger.debug('No numerical features for scaling')
        return None
    if params is None:
        logger.info(f'Getting default params for step {step_name}')
        return AUTO_CREATE_STEP_MAPPING[step_name](data)
    else:
        steps = get_steps_from_params(step_name, params)
        return steps


def universal_step_creating(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    if params is None:
        logger.info(f'Getting default params for step {step_name}')
        return AUTO_CREATE_STEP_MAPPING[step_name](data)
    else:
        steps = get_steps_from_params(step_name, params)
        return steps


RESOLVE_STEP_MAPPING = {
    PreprocessingStepEnum.imputation: get_imputation_step,
    PreprocessingStepEnum.scaling: get_scaling_step
}


def get_optional_steps(step_name: PreprocessingStepEnum,
                       data: TensorData,
                       params=None) -> PreprocessingStep:
    logger.info(f'Creating optional step {step_name}')
    if step_name in RESOLVE_STEP_MAPPING:
        step = RESOLVE_STEP_MAPPING[step_name](step_name, data, params)
    else:
        step = universal_step_creating(step_name, data, params)
    return step


def build_optional_plan(data: TensorData, optional_steps=None) -> PreprocessingPlan:

    optional_plan = PreprocessingPlan()

    for step_name in optional_steps.keys():
        step = get_optional_steps(step_name, data, optional_steps[step_name])
        optional_plan.add_step(step)
    return optional_plan
