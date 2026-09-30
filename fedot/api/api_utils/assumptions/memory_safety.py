"""Conservative candidate selection for unusually large multiclass fits."""

from typing import List, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.constants import MIN_NUMBER_OF_GENERATIONS
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


# Multiclass tree training can hold per-class gradients, histograms and many
# fitted trees at once. This is a *workload* threshold, not a dataset allowlist.
MAX_MULTICLASS_TREE_WORK = 1_000_000_000
HIGH_RESOURCE_CLASSIFIERS = {'catboost', 'xgboost', 'rf', 'extra_trees', 'knn'}


def memory_safe_operations(data: InputData, available_operations: Optional[List[str]]) -> Optional[List[str]]:
    """Remove models with large multiclass working sets when a light tree remains.

    Preserve the other preprocessing and modeling operations and the requested
    search/tuning budget. Explicit initial assumptions are handled by the caller.
    """
    if (not isinstance(data, InputData) or data.data_type != DataTypesEnum.table or
            data.task.task_type != TaskTypesEnum.classification or not available_operations or
            'lgbm' not in available_operations or data.target is None):
        return available_operations

    n_classes = len(pd.unique(np.asarray(data.target).ravel()))
    if n_classes <= 2 or len(data.features) * data.features.shape[1] * n_classes < MAX_MULTICLASS_TREE_WORK:
        return available_operations

    return [operation for operation in available_operations if operation not in HIGH_RESOURCE_CLASSIFIERS]


def bounded_population_size(population: int, n_jobs: int, composing_seconds: float,
                            estimated_fit_seconds: float) -> int:
    """Keep a few evolutionary generations feasible on a memory-limited workload.

    The initial pipeline is timed before the search starts. Allow headroom for
    slower mutations while preserving the user-provided time and tuning budgets.
    """
    if estimated_fit_seconds <= 0 or composing_seconds <= 0:
        return population
    feasible = int(0.8 * composing_seconds * n_jobs / (MIN_NUMBER_OF_GENERATIONS * estimated_fit_seconds))
    return min(population, max(2, feasible))
