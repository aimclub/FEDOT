from typing import Callable, Union, Sequence, Optional

from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.composer.metrics import Metric
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, RegressionMetricsEnum, MetricType,
                                                              MetricsEnum, TimeSeriesForecastingMetricsEnum)
from fedot.core.repository.tasks import Task
from fedot.utilities.define_metric_by_task import MetricByTask


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    _metrics_dict = {
        'accuracy': ClassificationMetricsEnum.accuracy,
        'roc_auc': ClassificationMetricsEnum.ROCAUC,
        'f1': ClassificationMetricsEnum.f1,
        'logloss': ClassificationMetricsEnum.logloss,
        'mae': RegressionMetricsEnum.MAE,
        'mse': RegressionMetricsEnum.MSE,
        'msle': RegressionMetricsEnum.MSLE,
        'mape': RegressionMetricsEnum.MAPE,
        'mase': TimeSeriesForecastingMetricsEnum.MASE,
        'smape': RegressionMetricsEnum.SMAPE,
        'r2': RegressionMetricsEnum.R2,
        'rmse': RegressionMetricsEnum.RMSE,
        'rmse_pen': RegressionMetricsEnum.RMSE_penalty,
        'silhouette': ClusteringMetricsEnum.silhouette,
        'node_number': ComplexityMetricsEnum.node_number
    }

    def __init__(self, task: Task, metrics: Optional[Union[str, MetricsEnum, Callable, Sequence]]):
        self.task: Task = task
        self.metric_functions: Sequence[MetricType] = self.obtain_metrics(metrics)

    @property
    def metric_names(self):
        return ApiMetrics.get_metric_names(self.metric_functions)

    @staticmethod
    def get_metric_names(metrics: Union[MetricType, Sequence[MetricType]]) -> Sequence[str]:
        return [str(metric) for metric in ensure_wrapped_in_sequence(metrics)]

    @staticmethod
    def get_metrics_mapping(metric_name: Union[str, Callable]) -> Union[Metric, Callable]:
        if isinstance(metric_name, Callable):
            # for custom metric
            return metric_name
        return ApiMetrics._metrics_dict[metric_name]

    def obtain_metrics(self, metrics: Optional[Union[str, MetricsEnum, Callable, Sequence]]) -> Sequence[MetricType]:
        """Chooses metric to use for quality assessment of pipeline during composition"""
        if metrics is None:
            metrics = MetricByTask.get_default_quality_metrics(self.task.task_type)

        metric_ids = []
        for specific_metric in ensure_wrapped_in_sequence(metrics):
            metric = None
            if isinstance(specific_metric, (str, Callable)):
                # metric was defined by name (str) or metric is a custom function
                metric = ApiMetrics.get_metrics_mapping(metric_name=specific_metric)
            elif isinstance(specific_metric, MetricsEnum):
                metric = specific_metric
            if metric is None:
                raise ValueError(f'Incorrect metric {specific_metric}')
            metric_ids.append(metric)
        return metric_ids
