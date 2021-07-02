from typing import Callable, Tuple, Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.validation.split import ts_cv_generator


def ts_metric_calculation(reference_data: InputData, cv_folds: int = 3,
                          metrics: [str, Callable] = None,
                          chain: Optional = None) -> Tuple[float, ...]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation
    """

    evaluated_metrics = [[] for _ in range(len(metrics))]

    for train_data, test_data in ts_cv_generator(reference_data, chain.log, cv_folds):
        chain.fit(train_data)

        for index, metric in enumerate(metrics):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            evaluated_metrics[index] += [metric_func(chain, reference_data=test_data)]

    evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
    return evaluated_metrics
