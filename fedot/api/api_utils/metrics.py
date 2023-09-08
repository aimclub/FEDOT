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
        default_metrics = MetricByTask.get_default_quality_metrics(self.task.task_type)
        self.metric_functions = ensure_wrapped_in_sequence(metrics) if metrics else default_metrics

    @property
    def metric_names(self):
        return ApiMetrics.get_metric_names(self.metric_functions)

    @staticmethod
    def get_metric_names(metrics: Union[MetricType, Sequence[MetricType]]) -> Sequence[str]:
        return [str(metric) for metric in ensure_wrapped_in_sequence(metrics)]
