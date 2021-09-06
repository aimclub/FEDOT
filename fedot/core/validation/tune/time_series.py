import numpy as np

from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.data.data import InputData
from fedot.core.validation.split import ts_cv_generator


def cv_time_series_predictions(pipeline, reference_data: InputData, log,
                               cv_folds: int, validation_blocks=None):
    """ Provide K-fold cross validation for time series with using in-sample
    forecasting on each step (fold)
    """

    # Place where predictions and actual values will be loaded
    predictions = []
    targets = []
    for train_data, test_data, vb_number in ts_cv_generator(reference_data, cv_folds, validation_blocks, log):
        if vb_number is None:
            # One fold validation
            pipeline.fit_from_scratch(train_data)
            output_pred = pipeline.predict(test_data)
            predictions = output_pred.predict
            targets = output_pred.target
            break
        else:
            # Cross validation: get number of validation blocks per each fold
            horizon = test_data.task.task_params.forecast_length * vb_number

            pipeline.fit_from_scratch(train_data)

            predicted_values = in_sample_ts_forecast(pipeline=pipeline,
                                                     input_data=test_data,
                                                     horizon=horizon)
            # Clip actual data by the forecast horizon length
            actual_values = test_data.target[-horizon:]
            predictions.extend(predicted_values)
            targets.extend(actual_values)

    predictions, targets = np.ravel(np.array(predictions)), np.ravel(np.array(targets))
    return predictions, targets
