from copy import deepcopy
from typing import Callable

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.validation.split import tabular_cv_generator, ts_cv_generator


def calculate_loss_function(loss_function, real_data, pred_data):
    """ Function processing preds and calculating metric (loss function)

    :param loss_function: loss function
    :param real_data: ground truth for evaluation
    :param pred_data: prediction for evaluation

    :return: calculated loss_function
    """
    try:
        # actual for regression and classification metrics that requires all classes probabilities
        metric_value = loss_function(real_data, pred_data)
    except ValueError:
        try:
            # change shape for ts prediction
            if real_data.task.task_type is TaskTypesEnum.ts_forecasting:
                pred_data_copy = deepcopy(pred_data)
                pred_data_copy.predict = np.ravel(pred_data_copy.predict)
                metric_value = loss_function(real_data, pred_data_copy)
            else:
                # transform 1st class probability to assigned class, actual for accuracy-like metrics with binary
                pred_data_copy = deepcopy(pred_data)
                pred_data_copy.predict = pred_data_copy.predict.round()
                metric_value = loss_function(real_data, pred_data_copy)
        except ValueError:
            # transform class probabilities to assigned class, actual for accuracy-like metrics with multiclass
            pred_data_copy = deepcopy(pred_data)
            pred_data_copy.predict = np.argmax(pred_data_copy.predict, axis=1)
            metric_value = loss_function(real_data, pred_data_copy)

    return metric_value


def cv_tabular_predictions(pipeline, reference_data: InputData, cv_folds: int, loss_function: Callable):
    """ Provide K-fold cross validation for tabular data"""

    metric_value = 0

    for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(test_data)
        metric_value += calculate_loss_function(loss_function, test_data, predicted_values)

    return metric_value / cv_folds


def cv_time_series_predictions(pipeline, reference_data: InputData, log,
                               cv_folds: int, loss_function: Callable, validation_blocks=None):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """

    # Place where predictions and actual values will be loaded
    metric_value = 0
    iter = 0
    for train_data, test_data in ts_cv_generator(reference_data, cv_folds, validation_blocks, log):
        if validation_blocks is None:
            # One fold validation
            pipeline.fit_from_scratch(train_data)
            output_pred = pipeline.predict(test_data)
            metric_value += loss_function(test_data, output_pred)
        else:
            # Cross validation: get number of validation blocks per each fold
            horizon = test_data.task.task_params.forecast_length * validation_blocks

            pipeline.fit_from_scratch(train_data)

            predicted_values = in_sample_ts_forecast(pipeline=pipeline,
                                                     input_data=test_data,
                                                     horizon=horizon)
            # Clip actual data by the forecast horizon length
            actual_values = test_data.target[-horizon:]
            test_data.target = actual_values
            predicted = OutputData(idx=np.arange(actual_values.shape[0]), features=test_data.features,
                                   predict=predicted_values, task=Task(TaskTypesEnum.ts_forecasting),
                                   data_type=DataTypesEnum.ts)
            metric_value += loss_function(test_data, predicted)
        iter += 1

    return metric_value / iter