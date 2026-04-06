import torch

from fedot.preprocessing.auto_create_step import AUTO_CREATE_STEP_MAPPING
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.preprocessor_types import PreprocessingStep, PreprocessingStepEnum
from fedot.preprocessing.structure import PipelineStructureExplorer


def get_steps_from_params(step_name: PreprocessingStepEnum, params):
    steps = []
    for step_params in params:
        step = PreprocessingStep(step_name, step_params['method'], step_params['features_idx'])
        if step_params['step_args'] is not None:
            step.step_args = step_params['step_args']
        steps.append(step)
    return steps


def has_nan_func(features: torch.Tensor) -> bool:
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


def is_imputation_needed(features: torch.Tensor, pipeline) -> bool:
    has_nan = has_nan_func(features)
    if pipeline is None:
        return has_nan
    has_imputation_operation = PipelineStructureExplorer.check_structure_by_tag(
            pipeline, tag_to_check='imputation')
    return has_nan and not has_imputation_operation


def get_optional_steps(step_name: PreprocessingStepEnum, data: TensorData, pipeline=None, params=None) -> PreprocessingStep:
    if is_imputation_needed(data.features, pipeline):
        if params is None:
            return AUTO_CREATE_STEP_MAPPING[step_name](data)

        else:
            steps = get_steps_from_params(step_name, params)
            return steps
    else:
        return None
    