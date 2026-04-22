import torch

from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStep, 
                                                    PreprocessingStepEnum,
                                                    ImputationMethodEnum,
                                                    ScalingMethodEnum,
                                                    FilteringMethodEnum)
from fedot.core.data.tensordata import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum


def find_nan_idx(features: torch.Tensor):
    nan_mask = torch.isnan(features)

    if features.ndim == 2:
        # (samples, features)
        cols_with_nan = nan_mask.any(dim=0)

    elif features.ndim == 3:
        # (samples, features, channels)
        cols_with_nan = nan_mask.any(dim=(0, 2))

    else:
        raise ValueError(f"Unsupported tensor shape: {features.shape}")

    indices = torch.where(cols_with_nan)[0]
    return indices.tolist()


def auto_imputation_steps(data: TensorData):

    nan_idx = find_nan_idx(data.features)

    if data.data_type == DataTypesEnum.tabular:
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

    else:
        step = PreprocessingStep(PreprocessingStepEnum.imputation, ImputationMethodEnum.ts_mean, nan_idx)
        return [step]


def auto_scaling_steps(data: TensorData):
    steps = []
    if len(data.numerical_idx) > 0:
        if data.data_type == DataTypesEnum.tabular:
            step = PreprocessingStep(PreprocessingStepEnum.scaling,
                                    ScalingMethodEnum.min_max,
                                    data.numerical_idx)
        else:
            step = PreprocessingStep(PreprocessingStepEnum.scaling,
                                    ScalingMethodEnum.seasonal,
                                    step_args={
                                        'period': 5
                                    })
        steps.append(step)
    else:
        steps = None
    return steps


def auto_clipping_step(data: TensorData):
    steps = []
    if len(data.numerical_idx) > 0:
        step = PreprocessingStep(PreprocessingStepEnum.filtering,
                                 FilteringMethodEnum.quantile,
                                 data.numerical_idx)
        steps.append(step)
    else:
        step = None
    return steps


AUTO_CREATE_STEP_MAPPING = {
    PreprocessingStepEnum.imputation: auto_imputation_steps,
    PreprocessingStepEnum.scaling: auto_scaling_steps,
    PreprocessingStepEnum.filtering: auto_clipping_step
}