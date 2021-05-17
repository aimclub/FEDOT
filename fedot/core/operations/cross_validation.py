from typing import Callable, Tuple, Optional
import numpy as np

from fedot.core.data.data import InputData, train_test_cv_generator
from fedot.core.chains.chain import Chain
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository


def cross_validation(reference_data: InputData, cv_folds: int,
                     metrics: [str, Callable], chain: Optional[Chain]) -> Tuple[float, ...]:
    if reference_data.task.task_type in [TaskTypesEnum.ts_forecasting, TaskTypesEnum.clustering]:
        raise NotImplementedError(f"Cross validation for {reference_data.task.task_type} is not implemented.")

    evaluated_metrics = [[] for _ in range(len(metrics))]

    for train_data, test_data in train_test_cv_generator(reference_data, cv_folds):
        chain.fit(train_data)

        for index, metric in enumerate(metrics):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            evaluated_metrics[index] += [metric_func(chain, reference_data=test_data)]

    evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))

    return evaluated_metrics
