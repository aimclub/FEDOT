from typing import List

from fedot.core.composer.metrics import RMSE, ROCAUC, Silhouette
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.quality_metrics_repository import MetricType
from fedot.core.repository.tasks import TaskTypesEnum


class MetricByTask:
    __metric_by_task = {TaskTypesEnum.regression: RMSE,
                        TaskTypesEnum.classification: ROCAUC,
                        TaskTypesEnum.clustering: Silhouette,
                        TaskTypesEnum.ts_forecasting: RMSE,
                        }

    def __init__(self, task_type):
        self.task_type = task_type
        self.metric_cls = self.__metric_by_task.get(task_type)

    def get_default_quality_metrics(self) -> List[MetricType]:
        if self.task_type == TaskTypesEnum.classification:
            return [self.metric_cls.get_value_with_penalty]
        return [self.metric_cls.get_value]

    def get_value(self, true: InputData, predicted: OutputData, round_up_to: int = 6):
        """Returns the value of metric defined by task"""
        try:
            return round(self.metric_cls.metric(reference=true, predicted=predicted), round_up_to)
        except ValueError:
            return self.metric_cls.default_value
