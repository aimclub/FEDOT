import logging
from enum import Enum
from typing import Dict, List, Union

import numpy as np

from fedot.industrial.core.metrics.metrics_implementation import (
    F1,
    MAE,
    MAPE,
    MSE,
    R2,
    RMSE,
    ROCAUC,
    Accuracy,
    Logloss,
    Precision,
)


class Metrics(Enum):
    roc_auc = ROCAUC
    f1 = F1
    precision = Precision
    accuracy = Accuracy
    logloss = Logloss
    rmse = RMSE
    r2 = R2
    mae = MAE
    mse = MSE
    mape = MAPE


DEF_METRIC_LIST = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']


class PerformanceAnalyzer:
    """Class responsible for calculating metrics for predictions.

    """

    def __init__(self):
        self.logger = logging.getLogger('PerformanceAnalyzer')

    def calculate_metrics(self,
                          target: Union[np.ndarray, List],
                          predicted_labels: Union[np.ndarray, list] = None,
                          predicted_probs: np.ndarray = None,
                          target_metrics: list = None) -> Dict:
        self.logger.info(f'Calculating metrics: {target_metrics}')
        try:
            if not isinstance(target[0], float):
                labels_diff = max(target) - max(predicted_labels)

                if min(predicted_labels) != min(target):
                    if min(target) == -1:
                        np.place(predicted_labels, predicted_labels == 1, [-1])
                        np.place(predicted_labels, predicted_labels == 0, [1])

                if labels_diff > 0:
                    predicted_labels = predicted_labels + abs(labels_diff)
                else:
                    target = target + abs(labels_diff)
        except Exception:
            target = target

        if target_metrics is not None:
            metric_list = target_metrics
        else:
            metric_list = DEF_METRIC_LIST

        result_metric = []
        for metric_name in metric_list:
            chosen_metric = Metrics[metric_name].value
            try:
                score = chosen_metric(target=target,
                                      predicted_labels=predicted_labels,
                                      predicted_probs=predicted_probs).metric()
                score = round(score, 3)
                result_metric.append(score)
            except Exception as err:
                self.logger.info(
                    f'Score cannot be calculated for {metric_name} metric')
                self.logger.info(err)
                result_metric.append(0)

        result_dict = dict(zip(metric_list, result_metric))
        self.logger.info(f'Metrics are: {result_dict}')

        return result_dict
