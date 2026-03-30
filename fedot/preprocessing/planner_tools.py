import torch
from typing import Optional, Dict, List

from golem.core.dag.convert import graph_structure_as_nx_graph

from fedot.core.data.tensordata import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME, PipelineStructureExplorer
from fedot.preprocessing.preprocessor_types import PreprocessingStep, ImputationMethodEnum, PreprocessingStepEnum

def data_type_is_tabular(data: TensorData) -> bool:
    return data.data_type is DataTypesEnum.tabular


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


def get_nan_columns(features: torch.Tensor):
    """
    """
    nan_mask = torch.isnan(features)
    cols_with_nan = nan_mask.any(dim=0)
    indices = torch.where(cols_with_nan)[0]
    return indices


def preprocess_imputation_params(features: torch.Tensor, params: Optional[List]=None) -> PreprocessingStep:

    method = None

    if params is not None:
        steps = []
        for param in params:
                    
            method = param["method"]
            
            if "features_idx" in param.keys():
                features_idx = param["features_idx"]
                step = PreprocessingStep(PreprocessingStepEnum.imputation, method, features_idx)
            else:
                break

            steps.append(step)

        if len(steps) > 0:
            return steps
        
    # TODO: default method is simple
    method = ImputationMethodEnum.moda if method is None else method
    features_idx = get_nan_columns(features)
    step = PreprocessingStep(PreprocessingStepEnum.imputation, method, features_idx)

    return step


def get_imputation_step(features: torch.Tensor, pipeline=None, params=None) -> PreprocessingStep:
    if not is_imputation_needed(features, pipeline):
        return None
    
    imputation_step = preprocess_imputation_params(features, params)
    return imputation_step


def preprocess_optional_params(step_name: PreprocessingStepEnum, features: torch.Tensor, params: List) -> List[PreprocessingStep]:
    steps = []

    for param in params:
        step = PreprocessingStep(step_name, param['method'], param['features_idx'])
        steps.append(step)

    return steps


def get_optional_step(step_name: PreprocessingStepEnum, features: torch.Tensor, pipeline=None, params=None) -> PreprocessingStep:
    if step_name == PreprocessingStepEnum.imputation:
        return get_imputation_step(features, pipeline, params)
    
    steps = preprocess_optional_params(step_name,features, params)

    return steps
