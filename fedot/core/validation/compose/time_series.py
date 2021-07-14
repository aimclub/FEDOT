from typing import Callable, Tuple, Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.validation.split import ts_cv_generator


def ts_metric_calculation(reference_data: InputData, cv_folds: int = 3,
                          metrics: [str, Callable] = None,
                          pipeline: Optional = None) -> Tuple[float, ...]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation

    :param reference_data: InputData for validation
    :param cv_folds: number of folds to split data
    :param metrics: name of metric or callable object
    :param pipeline: Pipeline for validation
    """

    evaluated_metrics = [[] for _ in range(len(metrics))]

    for train_data, test_data in ts_cv_generator(reference_data, pipeline.log, cv_folds):
        # Exception will be handled at the next level (metric evaluation)
        try:
            pipeline.fit(train_data)
        except Exception:
            pass

        for index, metric in enumerate(metrics):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            evaluated_metrics[index] += [metric_func(pipeline, reference_data=test_data)]

    evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
    return evaluated_metrics
