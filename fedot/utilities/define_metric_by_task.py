import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score as roc_auc

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


class TunerMetricByTask:

    def __init__(self, task_type):
        self.task_type = task_type

    def get_metric_and_params(self, input_data):
        """ Method return appropriate loss function for tuning

        :param input_data: InputData which will be used for training
        :return loss_function: function, which will calculate metric
        :return loss_params: parameters for loss function
        """
        if self.task_type == TaskTypesEnum.regression:
            # Default metric for regression
            loss_function = mean_squared_error
            loss_params = {'squared': False}
        elif self.task_type == TaskTypesEnum.ts_forecasting:
            # Default metric for time series forecasting
            loss_function = mean_squared_error
            loss_params = {'squared': False}
        elif self.task_type == TaskTypesEnum.classification:
            # Default metric for time classification
            amount_of_classes = len(np.unique(np.array(input_data.target)))
            if amount_of_classes == 2:
                # Binary classification
                loss_function = roc_auc
                loss_params = None
            else:
                # Metric for multiclass classification
                loss_function = roc_auc
                loss_params = {'multi_class': 'ovr'}
        else:
            raise NotImplementedError(f'Metric for "{self.task_type}" is not supported')
        return loss_function, loss_params
