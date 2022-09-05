from typing import Callable, Union, Sequence

from fedot.core.composer.metrics import Metric
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, RegressionMetricsEnum)


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    _task_dict = {
        'regression': ['rmse', 'mae'],
        'classification': ['roc_auc', 'f1'],
        'multiclassification': 'f1',
        'clustering': 'silhouette',
        'ts_forecasting': ['rmse', 'mae']
    }

    _metrics_dict = {
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

    def __init__(self, problem: str):
        if '/' in problem:
            # Solve multitask problem
            self.main_problem, self.side_problem = problem.split('/')
        else:
            self.main_problem = problem
            self.side_problem = None

    def get_problem_metrics(self) -> Union[str, Sequence[str]]:
        return ApiMetrics._task_dict[self.main_problem]

    @staticmethod
    def get_metrics_mapping(metric_name: Union[str, Callable]) -> Union[Metric, Callable]:
        if isinstance(metric_name, Callable):
            # for custom metric
            return metric_name
        return ApiMetrics._metrics_dict[metric_name]
