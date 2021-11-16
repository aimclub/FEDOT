from typing import Callable, List, Union

from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, RegressionMetricsEnum)


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    def __init__(self, problem: str):
        self.problem = problem

    def get_problem_metrics(self):
        task_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return task_dict[self.problem]

    def get_metrics_for_task(self, metric_name: Union[str, List[str]]):
        """ Return one metric for task by name (str)

        :param metric_name: names of metrics
        """
        task_metrics = self.get_problem_metrics()

        if type(metric_name) is not str:
            # Take only one metric (first) for optimisation
            metric_name = metric_name[0]

        composer_metric = self.get_composer_metrics_mapping(metric_name)
        tuner_metrics = self.get_tuner_metrics_mapping(metric_name)
        return task_metrics, composer_metric, tuner_metrics

    @staticmethod
    def get_tuner_metrics_mapping(metric_name):
        tuner_dict = {
            'acc': accuracy_score,
            'roc_auc': roc_auc_score,
            'f1': f1_score,
            'logloss': log_loss,
            'mae': mean_absolute_error,
            'mse': mean_squared_error,
            'r2': r2_score,
            'rmse': mean_squared_error,
        }

        return tuner_dict.get(metric_name)

    @staticmethod
    def get_composer_metrics_mapping(metric_name: Union[str, Callable]):
        if isinstance(metric_name, Callable):
            # for custom metric
            return metric_name

        composer_metric_dict = {
            'acc': ClassificationMetricsEnum.accuracy,
            'roc_auc': ClassificationMetricsEnum.ROCAUC,
            'f1': ClassificationMetricsEnum.f1,
            'logloss': ClassificationMetricsEnum.logloss,
            'mae': RegressionMetricsEnum.MAE,
            'mse': RegressionMetricsEnum.MSE,
            'msle': RegressionMetricsEnum.MSLE,
            'mape': RegressionMetricsEnum.MAPE,
            'r2': RegressionMetricsEnum.R2,
            'rmse': RegressionMetricsEnum.RMSE,
            'rmse_pen': RegressionMetricsEnum.RMSE_penalty,
            'silhouette': ClusteringMetricsEnum.silhouette,
            'node_num': ComplexityMetricsEnum.node_num
        }
        return composer_metric_dict[metric_name]
