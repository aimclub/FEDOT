import os
from typing import Callable, List, Union

from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, RegressionMetricsEnum)
from fedot.core.composer.metrics import smape, Metric


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    def __init__(self, problem: str):
        if '/' in problem:
            # Solve multitask problem
            self.main_problem, self.side_problem = problem.split('/')
        else:
            self.main_problem = problem
            self.side_problem = None

    def get_problem_metrics(self):
        task_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return task_dict[self.main_problem]

    @staticmethod
    def get_metrics_mapping(metric_name: Union[str, Callable]) -> Union[Metric, Callable]:
        if isinstance(metric_name, Callable):
            # for custom metric
            return metric_name

        metric_dict = {
            'acc': ClassificationMetricsEnum.accuracy,
            'roc_auc': ClassificationMetricsEnum.ROCAUC,
            'f1': ClassificationMetricsEnum.f1,
            'logloss': ClassificationMetricsEnum.logloss,
            'mae': RegressionMetricsEnum.MAE,
            'mse': RegressionMetricsEnum.MSE,
            'msle': RegressionMetricsEnum.MSLE,
            'mape': RegressionMetricsEnum.MAPE,
            'smape': RegressionMetricsEnum.SMAPE,
            'r2': RegressionMetricsEnum.R2,
            'rmse': RegressionMetricsEnum.RMSE,
            'rmse_pen': RegressionMetricsEnum.RMSE_penalty,
            'silhouette': ClusteringMetricsEnum.silhouette,
            'node_num': ComplexityMetricsEnum.node_num
        }
        return metric_dict[metric_name]
