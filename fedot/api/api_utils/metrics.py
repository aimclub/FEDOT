import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from typing import List

from sklearn.preprocessing import LabelBinarizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, MetricsRepository,
                                                              RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import probs_to_labels


class ApiMetricsHelper():

    def get_tuner_metrics_mapping(self,
                                  metric_name):
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

    def get_problem_metrics(self,
                            problem: str):
        task_dict = {
            'regression': ['rmse', 'mae'],
            'classification': ['roc_auc', 'f1'],
            'multiclassification': 'f1',
            'clustering': 'silhouette',
            'ts_forecasting': ['rmse', 'mae']
        }
        return task_dict[problem]

    def get_composer_metrics_mapping(self,
                                     metric_name: str):
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

    def get_metrics_for_task(self,
                             problem: str,
                             metric_name: str):
        task_metrics = self.get_problem_metrics(problem)
        composer_metric = self.get_composer_metrics_mapping(metric_name[0])
        tuner_metrics = self.get_tuner_metrics_mapping(metric_name[0])
        return task_metrics, composer_metric, tuner_metrics

    def check_prediction_shape(self,
                               task: Task,
                               metric_name: str,
                               real: InputData,
                               prediction: OutputData):
        if task == TaskTypesEnum.ts_forecasting:
            real.target = real.target[~np.isnan(prediction.predict)]
            prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

        if metric_name == 'roc_auc' and len(prediction.predict.shape) == 1:
            if real.num_classes == 2:
                prediction.predict = probs_to_labels(prediction.predict)
            else:
                real.target, prediction.predict = self.multiclass_roc_auc_score(real.target,
                                                                                prediction.predict)
        elif metric_name == 'f1' and len(prediction.predict.shape) > len(real.target.shape):
            prediction.predict = probs_to_labels(prediction.predict)
        else:
            pass

        return real.target, prediction.predict

    def multiclass_roc_auc_score(self,
                                 truth: List,
                                 pred: List):
        lb = LabelBinarizer()
        lb.fit(truth)
        truth = lb.transform(truth)
        pred = lb.transform(pred)
        return truth, pred
