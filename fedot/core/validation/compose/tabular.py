from typing import Callable, Optional, Tuple

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator


def table_cross_validation(reference_data: InputData, cv_folds: int,
                           metrics: [str, Callable], pipeline: Optional[Pipeline]) -> Tuple[float, ...]:
    """ Perform cross validation on tabular data for regression and classification tasks

    :param reference_data:
    :param cv_folds: number of folds to split data
    :param metrics: name of metric or callable object
    :param pipeline: Pipeline for validation
    """
    if reference_data.task.task_type is TaskTypesEnum.clustering:
        raise NotImplementedError(f"Tabular cross validation for {reference_data.task.task_type} is not supported")

    evaluated_metrics = [[] for _ in range(len(metrics))]

    for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
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
