from typing import Any, Dict, Iterable, List, Mapping

import torch

from fedot.core.backend.backend import Backend
from fedot.core.data.common.types import ArrayType, IndexType
from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStep,
                                                          PreprocessingStepEnum,
                                                          ImputationMethodEnum,
                                                          ScalingMethodEnum,
                                                          FilteringMethodEnum,
                                                          EncodingMethodEnum)
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum


def choose_label_ohe_plan(categorical_stats: Iterable[Mapping[str, Any]],
                          n_rows: int,
                          max_ohe_width: int = 500,
                          max_ohe_width_ratio: float = 0.5,
                          force_ohe_max_unique: int = 10,
                          use_drop_first: bool = False) -> Dict[Any, EncodingMethodEnum]:
    """Choose label or one-hot encoding for each categorical feature.

    Args:
        categorical_stats: Per-column statistics with `column`, `n_unique`, and
            optional `effective_unique` values.
        n_rows: Number of rows in the dataset.
        max_ohe_width: Absolute upper bound for total OHE-generated columns.
        max_ohe_width_ratio: Relative upper bound for OHE width as a fraction of
            the row count.
        force_ohe_max_unique: Prefer OHE for columns with cardinality not higher
            than this threshold while the global OHE budget allows it.
        use_drop_first: Estimate OHE cost as `n_categories - 1`. The current
            FEDOT OHE implementation keeps all categories, so automatic calls
            use `False`.

    Returns:
        Mapping from column identifiers to encoding methods.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")

    if max_ohe_width < 0:
        raise ValueError("max_ohe_width must be non-negative")

    if max_ohe_width_ratio < 0:
        raise ValueError("max_ohe_width_ratio must be non-negative")

    ohe_budget = min(
        max_ohe_width,
        int(max_ohe_width_ratio * n_rows),
    )

    candidates = []

    for stat in categorical_stats:
        column = stat["column"]
        n_unique = int(stat.get("effective_unique", stat["n_unique"]))

        if n_unique <= 1:
            ohe_cost = 0 if use_drop_first else n_unique
        elif use_drop_first:
            ohe_cost = n_unique - 1
        else:
            ohe_cost = n_unique

        candidates.append(
            {
                "column": column,
                "n_unique": n_unique,
                "ohe_cost": ohe_cost,
            }
        )

    candidates.sort(key=lambda item: item["ohe_cost"])

    used_ohe_width = 0
    plan: Dict[Any, EncodingMethodEnum] = {}

    for item in candidates:
        column = item["column"]
        n_unique = item["n_unique"]
        ohe_cost = item["ohe_cost"]

        fits_budget = used_ohe_width + ohe_cost <= ohe_budget

        if n_unique <= force_ohe_max_unique and fits_budget:
            plan[column] = EncodingMethodEnum.ohe
            used_ohe_width += ohe_cost
            continue

        if fits_budget:
            plan[column] = EncodingMethodEnum.ohe
            used_ohe_width += ohe_cost
        else:
            plan[column] = EncodingMethodEnum.label

    return plan


def _categorical_stats(features: ArrayType, categorical_idx: IndexType) -> List[Dict[str, int]]:
    """Calculate cardinality for selected categorical features."""
    pd_backend = Backend().pd

    stats = []
    for column_id in categorical_idx:
        series = pd_backend.Series(features[:, column_id])
        n_unique = int(series.dropna().nunique())
        stats.append({
            "column": column_id,
            "n_unique": n_unique,
        })

    return stats


def auto_encoding_steps(features: ArrayType, categorical_idx: IndexType) -> List[PreprocessingStep]:
    """Build automatic encoding steps for uncovered categorical columns.

    The selected per-column methods are grouped into preprocessing steps because
    the execution pipeline applies one handler per step.
    """
    if categorical_idx is None or len(categorical_idx) == 0:
        return []

    stats = _categorical_stats(features, categorical_idx)
    encoding_plan = choose_label_ohe_plan(stats, n_rows=features.shape[0])

    steps = []
    for method in (EncodingMethodEnum.label, EncodingMethodEnum.ohe):
        features_idx = [
            column
            for column in categorical_idx
            if encoding_plan[column] == method
        ]
        if len(features_idx) == 0:
            continue

        steps.append(PreprocessingStep(
            step=PreprocessingStepEnum.encoding,
            method=method,
            features_idx=features_idx,
        ))

    return steps


def find_nan_idx(features: torch.Tensor):
    """Find feature indices containing at least one missing value.

    Args:
        features: Input feature tensor. Supported shapes are
            `(n_samples, n_features)` and `(n_samples, n_features, n_channels)`.

    Returns:
        List of feature indices where at least one `NaN` is present.
    """
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
    """Build default imputation steps based on data type and missing values.

    For tabular data, missing values are split by feature type:
    categorical columns use mode imputation, numerical columns use median
    imputation, and remaining columns fall back to row deletion.
    For time series data, a single time-series mean imputation step is created.

    Args:
        data: TensorData object with detected feature type indices.

    Returns:
        List of automatically created imputation preprocessing steps.
    """
    nan_idx = find_nan_idx(data.features)

    if data.data_type == DataTypesEnum.tabular:
        steps = []

        if len(data.categorical_idx) > 0:
            cat_nan_idx = list(set(nan_idx) & set(data.categorical_idx))
            if len(cat_nan_idx) > 0:
                step = PreprocessingStep(
                    PreprocessingStepEnum.imputation, ImputationMethodEnum.mode, cat_nan_idx)
                steps.append(step)
        else:
            cat_nan_idx = []

        if len(data.numerical_idx) > 0:
            num_nan_idx = list(set(nan_idx) & set(data.numerical_idx))
            if len(num_nan_idx) > 0:
                step = PreprocessingStep(
                    PreprocessingStepEnum.imputation, ImputationMethodEnum.median, num_nan_idx)
                steps.append(step)
        else:
            num_nan_idx = []

        remain = list(set(nan_idx) - set(cat_nan_idx) - set(num_nan_idx))
        if len(remain) > 0:
            step = PreprocessingStep(
                PreprocessingStepEnum.imputation, ImputationMethodEnum.delete_raw, remain)
            steps.append(step)

        return steps

    else:
        step = PreprocessingStep(
            PreprocessingStepEnum.imputation, ImputationMethodEnum.ts_mean, nan_idx)
        return [step]


def auto_scaling_steps(data: TensorData):
    """Build default scaling steps for numerical features.

    Uses min-max scaling for tabular data and seasonal scaling for time series
    data when numerical features are available.

    Args:
        data: TensorData object with `numerical_idx` and `data_type`.

    Returns:
        List with scaling step(s), or `None` when no numerical features exist.
    """
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
    """Build default quantile clipping step for numerical features.

    Args:
        data: TensorData object with `numerical_idx`.

    Returns:
        List with one clipping step when numerical features exist; otherwise an
        empty list.
    """
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
