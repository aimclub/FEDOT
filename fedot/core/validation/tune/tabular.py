import numpy as np

from fedot.core.validation.split import tabular_cv_generator
from fedot.core.data.data import InputData


def cv_tabular_predictions(pipeline, reference_data: InputData, cv_folds: int):
    """ Provide K-fold cross validation for tabular data"""

    predictions = []
    targets = []

    for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(test_data).predict
        actual_values = test_data.target
        predictions.extend(predicted_values)
        targets.extend(actual_values)

    if train_data.num_classes <= 2:
        predictions, targets = np.ravel(np.array(predictions)), np.ravel(np.array(targets))

    return predictions, targets
