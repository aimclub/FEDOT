from typing import Callable

from fedot.core.data.data import InputData
from fedot.core.pipelines.tuning.tuner_interface import _calculate_loss_function
from fedot.core.validation.split import tabular_cv_generator


def cv_tabular_predictions(pipeline, reference_data: InputData, cv_folds: int, loss_function: Callable):
    """ Provide K-fold cross validation for tabular data"""

    metric_value = 0

    for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(test_data)
        metric_value += _calculate_loss_function(loss_function, test_data, predicted_values)

    return metric_value / cv_folds
