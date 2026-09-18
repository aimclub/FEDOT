import logging
from enum import Enum
from typing import List

from fedot.preprocessing.planner.auto_create_step import AUTO_CREATE_STEP_MAPPING
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import get_idx_from_features_names
from fedot.preprocessing.tools.preprocessor_types import (
    PreprocessingStep,
    PreprocessingStepEnum,
    ScalingMethodEnum,
)
from fedot.preprocessing.planner.planner import PreprocessingPlan
from fedot.preprocessing.tools.index_mapping_tools import update_indices


logger = logging.getLogger(__name__)


def _all_feature_indices(data: TensorData) -> List[int]:
    """Return indices of all feature columns for 2D/3D tensors."""
    features = data.features
    if features.ndim in (2, 3):
        return list(range(features.shape[1]))
    raise ValueError(f'Unsupported tensor shape: {features.shape}')


def get_steps_from_params(data: TensorData, step_name: PreprocessingStepEnum, params):
    """Convert user step parameters into preprocessing step objects.

    Explicit ``features_idx`` always produces steps, even when the current train
    batch does not need the transformation yet.

    Args:
        step_name: Optional preprocessing stage name.
        params: List of dictionaries describing methods, feature indices and
            optional `step_args` / custom implementation.

    Returns:
        List of constructed preprocessing steps for the given stage.
    """
    steps = []
    for step_params in params:
        features_idx = get_idx_from_features_names(
            step_params['features_idx'], data.features_names)
        features_idx = update_indices(data.idx_mapping, features_idx)
        step = PreprocessingStep(
            step_name, step_params['method'], features_idx)
        if step_params['step_args'] is not None:
            step.step_args = step_params['step_args']

        implementation = step_params.get('implementation')
        if implementation is not None:
            step.implementation = step_params['implementation']
        steps.append(step)
    return steps


def _default_features_for_step(step_name: PreprocessingStepEnum, data: TensorData) -> List[int]:
    """Select columns when the user requested a step without explicit indices.

    Imputation uses all feature columns. Scaling uses detected numerical columns.
    Column policy is driven by the step kind and TensorData metadata, not by a
    hard-coded catalogue of method names.
    """
    if step_name == PreprocessingStepEnum.scaling:
        return list(data.numerical_idx or [])
    return _all_feature_indices(data)


def _steps_from_method_only(step_name: PreprocessingStepEnum, data: TensorData, method):
    """Build steps when stage config is a single method; columns are selected automatically."""
    if not isinstance(method, (Enum, str)):
        return None

    features_idx = _default_features_for_step(step_name, data)
    if not features_idx:
        return None

    step = PreprocessingStep(step_name, method, features_idx)
    if step_name == PreprocessingStepEnum.scaling and method == ScalingMethodEnum.seasonal:
        step.step_args = {'period': 5}
    return [step]


def get_imputation_step(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    """Resolve imputation steps for optional preprocessing plan.

    When the user requests imputation, steps are created even if the current
    train batch has no missing values, so fitted handlers can impute NaNs that
    appear later at predict time.

    Args:
        step_name: Optional preprocessing stage name (`imputation`).
        data: Input tensor data used for automatic column selection.
        params: User-defined imputation strategy parameters, or `None` for
            automatic step creation. A single method enum/str is also accepted;
            then all feature columns are selected for that method.

    Returns:
        List of imputation steps, or `None` when no applicable columns exist.
    """
    if params is None:
        logger.info(f'Getting default params for step {step_name}')
        return AUTO_CREATE_STEP_MAPPING[step_name](data)
    method_only_steps = _steps_from_method_only(step_name, data, params)
    if method_only_steps is not None:
        return method_only_steps
    return get_steps_from_params(data, step_name, params)


def get_scaling_step(step_name: PreprocessingStepEnum, data: TensorData, params=None) -> PreprocessingStep:
    """Resolve scaling steps for optional preprocessing plan.

    Args:
        step_name: Optional preprocessing stage name (`scaling`).
        data: Input tensor data with feature type indices.
        params: User-defined scaling strategy parameters, or `None` for
            automatic step creation. A single method enum/str is also accepted;
            then numerical columns are selected automatically.

    Returns:
        List of scaling steps, or `None` when no numerical features exist.
    """
    if params is None:
        if len(data.numerical_idx or []) == 0:
            logger.debug('No numerical features for scaling')
            return None
        logger.info(f'Getting default params for step {step_name}')
        return AUTO_CREATE_STEP_MAPPING[step_name](data)
    method_only_steps = _steps_from_method_only(step_name, data, params)
    if method_only_steps is not None:
        return method_only_steps
    return get_steps_from_params(data, step_name, params)


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
    method_only_steps = _steps_from_method_only(step_name, data, params)
    if method_only_steps is not None:
        return method_only_steps
    return get_steps_from_params(data, step_name, params)


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
