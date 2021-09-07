import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from fedot.core.data.data import InputData, OutputData

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum, ClusteringMetricsEnum,
                                                              ComplexityMetricsEnum, MetricsRepository,
                                                              RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum

composer_metrics_mapping = {
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

tuner_metrics_mapping = {
    'acc': accuracy_score,
    'roc_auc': roc_auc_score,
    'f1': f1_score,
    'logloss': log_loss,
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score,
    'rmse': mean_squared_error,
}


def tuner_metric_by_name(metric_name: str, train_data: InputData, task: Task):
    """ Function allow to obtain metric for tuner by its name

    :param metric_name: name of metric
    :param train_data: InputData for train
    :param task: task to solve

    :return tuner_loss: loss function for tuner
    :return loss_params: parameters for tuner loss (can be None in some cases)
    """
    loss_params = None
    tuner_loss = tuner_metrics_mapping.get(metric_name)
    if tuner_loss is None:
        raise ValueError(f'Incorrect tuner metric {tuner_loss}')

    if metric_name == 'rmse':
        loss_params = {'squared': False}
    elif metric_name == 'roc_auc' and task == TaskTypesEnum.classification:
        amount_of_classes = len(np.unique(np.array(train_data.target)))
        if amount_of_classes == 2:
            # Binary classification
            loss_params = None
        else:
            # Metric for multiclass classification
            loss_params = {'multi_class': 'ovr'}
    return tuner_loss, loss_params


def _autodetect_data_type(task: Task) -> DataTypesEnum:
    if task.task_type == TaskTypesEnum.ts_forecasting:
        return DataTypesEnum.ts
    else:
        return DataTypesEnum.table


def save_predict(predicted_data: OutputData):
    if len(predicted_data.predict.shape) >= 2:
        prediction = predicted_data.predict.tolist()
    else:
        prediction = predicted_data.predict
    return pd.DataFrame({'Index': predicted_data.idx,
                         'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)


def array_to_input_data(features_array: np.array,
                        target_array: np.array,
                        task: Task = Task(TaskTypesEnum.classification)):
    data_type = _autodetect_data_type(task)
    idx = np.arange(len(features_array))

    return InputData(idx=idx, features=features_array, target=target_array, task=task, data_type=data_type)

