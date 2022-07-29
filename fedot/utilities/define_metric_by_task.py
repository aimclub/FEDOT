from fedot.core.composer.metrics import RMSE, ROCAUC, Silhouette
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum


class MetricByTask:
    __metric_by_task = {TaskTypesEnum.regression: RMSE,
                        TaskTypesEnum.classification: ROCAUC,
                        TaskTypesEnum.clustering: Silhouette,
                        TaskTypesEnum.ts_forecasting: RMSE,
                        }

    def __init__(self, task_type):
        self.metric_cls = self.__metric_by_task.get(task_type)

    def get_value(self, true: InputData, predicted: OutputData, round_up_to: int = 6):
        """Returns the value of metric defined by task"""
        try:
            return round(self.metric_cls.metric(reference=true, predicted=predicted), round_up_to)
        # TODO or raise ValueError? What to return in case of failure
        except ValueError:
            return 0.0
