from typing import Callable, Tuple, Optional

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.tune.time_series import ts_cross_validation


def ts_metric_calculation(reference_data: InputData, cv_folds: int,
                          metrics: [str, Callable], chain: Optional) -> Tuple[float, ...]:
    """ Determine metric value for chain based on data for validation """

    evaluated_metrics = [[] for _ in range(len(metrics))]
    # TODO implement data
