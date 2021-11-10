import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import probs_to_labels


class ApiMetrics:
    """
    Class for metrics matching. Handling both "metric name" - "metric instance"
    both for composer and tuner
    """

    def __init__(self, problem):
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

    def get_metrics_for_task(self, metric_name: str):

        task_metrics = self.get_problem_metrics()
        composer_metric = self.get_composer_metrics_mapping(metric_name[0])
        tuner_metrics = self.get_tuner_metrics_mapping(metric_name[0])
        return task_metrics, composer_metric, tuner_metrics

    def check_prediction_shape(self, task: Task, metric_name: str,
                               real: InputData,  prediction: OutputData):
        if task == TaskTypesEnum.ts_forecasting:
            real.target = real.target[~np.isnan(prediction.predict)]
            prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

        if metric_name == 'f1':
            if len(prediction.predict.shape) > len(real.target.shape):
                prediction.predict = probs_to_labels(prediction.predict)
            elif real.num_classes == 2:
                prediction.predict = probs_to_labels(self.convert_to_two_classes(prediction.predict))
        return real.target, prediction.predict

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
    def get_composer_metrics_mapping(metric_name: str):
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

    @staticmethod
    def convert_to_two_classes(predict):
        return np.vstack([1 - predict, predict]).transpose()
