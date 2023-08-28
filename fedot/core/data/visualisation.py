from os import PathLike
from typing import Optional, Any, Union

import matplotlib.pyplot as plt
import numpy as np

from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum


def plot_forecast(data: [InputData, MultiModalData], prediction: OutputData, in_sample: bool = False,
                  target: Optional[Any] = None, title: str = 'Forecast',
                  save_path: Optional[Union[PathLike, str]] = None):
    """
    Function for drawing plot with time series forecast. If data.target is None function plot prediction
    as future values. If not - we use last data features as validation.

    Args:
        data: the InputData or MultiModalData with actual time series as features
        prediction: the OutputData with predictions
        in_sample: if obtained prediction was in sample.
        save_path: path to save the visualized forecast
        title: custom title for the plot
        If ``False`` plots predictions as future values for test data features.
        target: user-specified name of target variable for MultiModalData
    """
    if isinstance(data, MultiModalData):
        if not target:
            raise AttributeError("Can't visualize. Target of MultiModalData not set.")
        data = data.extract_data_source(target)
    if data.is_multi_ts:
        actual_time_series = data.features[:, 0]
    else:
        actual_time_series = data.features
    target_time_series = data.target
    predict = prediction.predict
    pred_start = len(actual_time_series)
    if in_sample:
        pred_start = len(actual_time_series) - len(predict)
    elif target_time_series is not None:
        actual_time_series = np.concatenate([actual_time_series, target_time_series], axis=0)

    padding = min(pred_start, 72)

    first_idx = pred_start - padding

    plt.plot(np.arange(pred_start, pred_start + len(predict)),
             predict, label='Predicted', c='blue')
    plt.plot(np.arange(first_idx, len(actual_time_series)),
             actual_time_series[first_idx:], label='Actual values', c='green')

    # Plot black line which divide our array into train and test
    plt.plot([pred_start, pred_start],
             [min(actual_time_series[first_idx:]), max(actual_time_series[first_idx:])], c='black',
             linewidth=1)

    if in_sample:
        in_sample_steps = np.arange(pred_start, pred_start + len(predict), data.task.task_params.forecast_length)[1:]
        plt.vlines(in_sample_steps,
                   ymin=min(actual_time_series[first_idx:]),
                   ymax=max(actual_time_series[first_idx:]), color='red', linewidth=1)
    plt.legend(fontsize=15)
    plt.title(title)
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_biplot(prediction: OutputData):
    """
    Function for drawing biplot with regression task.

    :param prediction: the OutputData with prediction and target.
    Target should be not null
    """

    target = prediction.target
    predict = prediction.predict

    if target is None:
        raise ValueError('Target must be not None to plot biplot')

    plt.figure(figsize=(10, 10))
    plt.scatter(target, predict)
    plt.grid()
    # plot bisect
    min_coord = min(np.min(target), np.min(predict))
    max_coord = max(np.max(target), np.max(predict))
    min_max_bisect = [min_coord, max_coord]
    plt.plot(min_max_bisect, min_max_bisect, c='black', linewidth=1)

    plt.title("Biplot")
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.show()


def plot_roc_auc(data: InputData, prediction: OutputData):
    """
    Function for drawing roc curve for classification task.

    :param data: the InputData with validation data
    :param prediction: prediction for data
    """
    if data.num_classes == 2:
        fpr, tpr, threshold = ROCAUC.roc_curve(data.target, prediction.predict)
        roc_auc = ROCAUC.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='best')
    else:
        for i in range(data.num_classes):
            fpr, tpr, threshold = ROCAUC.roc_curve(data.target, prediction.predict[:, i],
                                                   pos_label=data.class_labels[i])
            roc_auc = ROCAUC.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'label-{i} AUC = %0.2f' % roc_auc)
            plt.legend(loc='best')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
