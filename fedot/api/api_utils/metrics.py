from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, MetricsRepository,
                                                              RegressionMetricsEnum)


class API_metrics_helper():

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
