import json
import os

import numpy as np

from examples.simple.regression.regression_with_tuning import get_regression_dataset
from examples.simple.regression.regression_pipelines import regression_ransac_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def create_correct_path(path: str, dirname_flag: bool = False):
    """
    Create path with time which was created during the testing process.
    """
    last_el = None
    for dirname in next(os.walk(os.path.curdir))[1]:
        if dirname.endswith(path):
            if dirname_flag:

                last_el = dirname
            else:
                file = os.path.join(dirname, path + '.json')
                last_el = file
    return last_el


def run_import_export_example(pipeline_path, pipeline):
    features_options = {'informative': 1, 'bias': 0.0}
    samples_amount = 100
    features_amount = 2
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options,
                                                              samples_amount,
                                                              features_amount)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    # Get pipeline and fit it
    pipeline.fit_from_scratch(train_input)

    predicted_output = pipeline.predict(predict_input)
    prediction_before_export = np.array(predicted_output.predict)
    print(f'Before export {prediction_before_export[:4]}')

    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(predict_input)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    print(f'After import {prediction_after_export[:4]}')

    dict_pipeline, dict_fitted_operations = pipeline.save()
    dict_pipeline = json.loads(dict_pipeline)
    pipeline_from_dict = Pipeline.from_serialized(dict_pipeline, dict_fitted_operations)

    predicted_output = pipeline_from_dict.predict(predict_input)
    prediction = np.array(predicted_output.predict)
    print(f'Prediction from pipeline loaded from dict {prediction[:4]}')


if __name__ == '__main__':
    run_import_export_example(pipeline_path='import_export', pipeline=regression_ransac_pipeline())
