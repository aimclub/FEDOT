import torch

from fedot.preprocessing.preprocessor_types import PreprocessingStep, PreprocessingStepEnum, ImputationMethodEnum
from fedot.core.data.tensordata import TensorData


def find_nan_idx(features: torch.Tensor):
    nan_mask = torch.isnan(features)
    cols_with_nan = nan_mask.any(dim=0)
    indices = torch.where(cols_with_nan)[0]
    return indices.tolist()


def auto_imputation_steps(data: TensorData):

    nan_idx = find_nan_idx(data.features)

    steps = []

    if len(data.categorical_idx) > 0:
        cat_nan_idx = list(set(nan_idx) & set(data.categorical_idx))
        step = PreprocessingStep(PreprocessingStepEnum.imputation, ImputationMethodEnum.mode, cat_nan_idx)
        steps.append(step)
    else:
        cat_nan_idx = []

    if len(data.numerical_idx) > 0:
        num_nan_idx = list(set(nan_idx) & set(data.numerical_idx))
        step = PreprocessingStep(PreprocessingStepEnum.imputation, ImputationMethodEnum.median, num_nan_idx)
        steps.append(step)
    else:
        num_nan_idx = []

    remain = list(set(nan_idx) - set(cat_nan_idx) - set(num_nan_idx))
    if len(remain) > 0:
        step = PreprocessingStep(PreprocessingStepEnum.imputation, ImputationMethodEnum.delete_raw, remain)
        steps.append(step)

    return steps


AUTO_CREATE_STEP_MAPPING = {
    PreprocessingStepEnum.imputation: auto_imputation_steps,
}