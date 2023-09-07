from typing import Callable, Union, Sequence, Optional

from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.repository.quality_metrics_repository import MetricType, MetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task
from fedot.utilities.define_metric_by_task import MetricByTask


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    def __init__(self, task: Task, metrics: Optional[Union[str, MetricsEnum, Callable, Sequence]]):
        self.task: Task = task
        self.metric_functions: Sequence[MetricType] = self.obtain_metrics(metrics)

    @property
    def metric_names(self):
        return ApiMetrics.get_metric_names(self.metric_functions)

    @staticmethod
    def get_metric_names(metrics: Union[MetricType, Sequence[MetricType]]) -> Sequence[str]:
        return [str(metric) for metric in ensure_wrapped_in_sequence(metrics)]

    def obtain_metrics(self, metrics: Optional[Union[str, MetricsEnum, Callable, Sequence]]) -> Sequence[MetricType]:
        """Chooses metric to use for quality assessment of pipeline during composition"""
        if metrics is None:
            metric_ids = MetricByTask.get_default_quality_metrics(self.task.task_type)
        else:
            metric_ids = []
            for specific_metric in ensure_wrapped_in_sequence(metrics):
                if isinstance(specific_metric, (Callable, MetricsEnum)):
                    # metric is a custom function
                    metric = specific_metric
                else:
                    # metric was defined by str
                    metric = MetricsRepository.metric_id_by_name(specific_metric)
                if metric is None:
                    raise ValueError(f'Incorrect metric {specific_metric}')
                metric_ids.append(metric)
        return metric_ids
