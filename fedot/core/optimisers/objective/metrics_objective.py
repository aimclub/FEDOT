from typing import Union, Iterable, Callable

from fedot.core.optimisers.objective import Objective
from fedot.core.repository.quality_metrics_repository import \
    MetricType, MetricsEnum, MetricsRepository, ComplexityMetricsEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence


class MetricsObjective(Objective):
    def __init__(self,
                 metrics: Union[MetricType, Iterable[MetricType]],
                 is_multi_objective: bool = False):
        quality_metrics = {}
        complexity_metrics = {}

        for metric in ensure_wrapped_in_sequence(metrics):
            if isinstance(metric, MetricsEnum):
                metric_func = MetricsRepository().metric_by_id(metric)

                if isinstance(metric, ComplexityMetricsEnum):
                    complexity_metrics[metric] = metric_func
                else:
                    quality_metrics[metric] = metric_func
            elif isinstance(metric, Callable):
                metric_id = str(metric)
                quality_metrics[metric_id] = metric

        super().__init__(quality_metrics, complexity_metrics, is_multi_objective)
