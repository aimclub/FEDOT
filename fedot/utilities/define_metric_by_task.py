from typing import List

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import (
    MetricsEnum,
    RegressionMetricsEnum,
    ClassificationMetricsEnum,
    ClusteringMetricsEnum,
    MetricsRepository
)


class MetricByTask:
    __metric_by_task = {TaskTypesEnum.regression: RegressionMetricsEnum.RMSE,
                        TaskTypesEnum.classification: ClassificationMetricsEnum.ROCAUC_penalty,
                        TaskTypesEnum.clustering: ClusteringMetricsEnum.silhouette,
                        TaskTypesEnum.ts_forecasting: RegressionMetricsEnum.RMSE,
                        }

    @staticmethod
    def get_default_quality_metrics(task_type: TaskTypesEnum) -> List[MetricsEnum]:
        return [MetricByTask.__metric_by_task.get(task_type)]

    @staticmethod
    def compute_default_metric(task_type: TaskTypesEnum, true: InputData, predicted: OutputData, round_up_to: int = 6) -> float:
        """Returns the value of metric defined by task"""
        metric_id = MetricByTask.get_default_quality_metrics(task_type)[0]
        metric = MetricsRepository.metric_class_by_id(metric_id)
        try:
            return round(metric.metric(reference=true, predicted=predicted), round_up_to)
        except ValueError:
            return metric.default_value
