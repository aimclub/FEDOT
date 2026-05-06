import torch
import logging

from fedot.preprocessing.planner.auto_create_step import AUTO_CREATE_STEP_MAPPING
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStep, PreprocessingStepEnum
from fedot.preprocessing.planner.planner import PreprocessingPlan


logger = logging.getLogger(__name__)


def get_steps_from_params(step_name: PreprocessingStepEnum, params):
    """Convert user step parameters into preprocessing step objects.

    Args:
        step_name: Optional preprocessing stage name.
        params: List of dictionaries describing methods, feature indices and
            optional `step_args` / custom implementation.

    Returns:
        List of constructed preprocessing steps for the given stage.
    """
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
    """Check whether feature tensor contains at least one missing value.

    Args:
        features: Input feature tensor.

    Returns:
        `True` if tensor contains at least one `NaN`, otherwise `False`.
    """
    return torch.isnan(features).any().item()


def get_imputation_step(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    """Resolve imputation steps for optional preprocessing plan.

    Args:
        step_name: Optional preprocessing stage name (`imputation`).
        data: Input tensor data used for automatic rule checks.
        params: User-defined imputation strategy parameters, or `None` for
            automatic step creation.

    Returns:
        List of imputation steps, or `None` when imputation is not required.
    """
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
    """Resolve scaling steps for optional preprocessing plan.

    Args:
        step_name: Optional preprocessing stage name (`scaling`).
        data: Input tensor data with feature type indices.
        params: User-defined scaling strategy parameters, or `None` for
            automatic step creation.

    Returns:
        List of scaling steps, or `None` when no numerical features exist.
    """
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
    """Resolve optional steps for stages without special conditions.

    Args:
        step_name: Optional preprocessing stage name.
        data: Input tensor data used for automatic step creation.
        params: User-defined stage parameters, or `None` for defaults.

    Returns:
        List of preprocessing steps for the requested stage.
    """
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
    """Create optional preprocessing steps for a single stage.

    Args:
        step_name: Optional preprocessing stage name.
        data: Input tensor data to analyze.
        params: Stage configuration parameters, or `None` for automatic mode.

    Returns:
        Stage step list (or `None`) resolved by stage-specific rules.
    """
    logger.info(f'Creating optional step {step_name}')
    if step_name in RESOLVE_STEP_MAPPING:
        step = RESOLVE_STEP_MAPPING[step_name](step_name, data, params)
    else:
        step = universal_step_creating(step_name, data, params)
    return step


def build_optional_plan(data: TensorData, optional_steps=None) -> PreprocessingPlan:
    """Build optional preprocessing plan from user strategy configuration.

    Args:
        data: Input tensor data used to derive automatic/default steps.
        optional_steps: Mapping from `PreprocessingStepEnum` to stage parameter
            list (or `None` for defaults per stage).

    Returns:
        Prepared optional preprocessing plan with resolved steps.
    """
    optional_plan = PreprocessingPlan()

    for step_name in optional_steps.keys():
        step = get_optional_steps(step_name, data, optional_steps[step_name])
        optional_plan.add_step(step)
    return optional_plan
