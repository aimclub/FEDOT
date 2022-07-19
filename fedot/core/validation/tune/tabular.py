import numpy as np
from typing import Callable

from fedot.core.pipelines.tuning.tuner_interface import _calculate_loss_function
from fedot.core.validation.split import tabular_cv_generator
from fedot.core.data.data import InputData


def cv_tabular_predictions(pipeline, reference_data: InputData, cv_folds: int, loss_function: Callable):
    """ Provide K-fold cross validation for tabular data"""

    metric_value = 0

    for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(test_data)
        metric_value += _calculate_loss_function(loss_function, test_data, predicted_values)

    return metric_value / cv_folds


import numpy as np
from typing import Callable

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.validation.split import ts_cv_generator


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

