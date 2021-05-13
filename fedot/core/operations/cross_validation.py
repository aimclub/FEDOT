from typing import Callable
import numpy as np

from fedot.core.data.data import InputData, train_test_data_generator
from fedot.core.chains.chain import Chain
from fedot.core.repository.quality_metrics_repository import MetricsRepository


def cross_validation(chain: Chain, reference_data: InputData, cv: int, metrics: [str, Callable]):
    evaluated_metrics = [[] for _ in range(len(metrics))]

    for train_data, test_data in train_test_data_generator(reference_data, cv):
        chain.fit(train_data)

        for index, metric in enumerate(metrics):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            evaluated_metrics[index] += [metric_func(chain, reference_data=test_data)]

    return list(map(lambda x: np.mean(x), evaluated_metrics))
