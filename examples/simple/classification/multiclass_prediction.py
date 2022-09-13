import datetime
import os
import random

import pandas as pd

from fedot.utilities.requirements_notificator import warn_requirement

try:
    import openpyxl
except ImportError:
    warn_requirement('openpyxl', 'fedot[examples]', should_raise=True)

from datetime import timedelta

import numpy as np

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import (
    ensure_directory_exists,
    fedot_project_root,
    get_split_data_paths,
    probs_to_labels,
    save_file_to_csv,
    split_data
)
from sklearn.metrics import roc_auc_score as roc_auc

random.seed(1)
np.random.seed(1)


def create_multi_clf_examples_from_excel(file_path: str, return_df: bool = False):
    """ Return dataframe from excel file or path to the csv file """
    df = pd.read_excel(file_path, engine='openpyxl')
    train, test = split_data(df)
    file_dir_name = file_path.replace('.', '/').split('/')[-2]
    file_csv_name = f'{file_dir_name}.csv'
    directory_names = ['examples', 'data', file_dir_name]

    # Check does obtained directory exist or not
    ensure_directory_exists(directory_names)
    if return_df:
        # Need to return dataframe and path to the file in csv format
        path = os.path.join(directory_names[0], directory_names[1], directory_names[2], file_csv_name)
        full_file_path = os.path.join(str(fedot_project_root()), path)
        save_file_to_csv(df, full_file_path)
        return df, full_file_path
    else:
        # Need to return only paths to the files with train and test data
        full_train_file_path, full_test_file_path = get_split_data_paths(directory_names)
        save_file_to_csv(train, full_train_file_path)
        save_file_to_csv(train, full_test_file_path)
        return full_train_file_path, full_test_file_path


def get_model(train_file_path: str, cur_lead_time: datetime.timedelta = timedelta(seconds=60)):
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)

    # the search of the models provided by the framework
    # that can be used as nodes in a pipeline for the selected task
    models_repo = OperationTypesRepository()
    available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = ClassificationMetricsEnum.ROCAUC_penalty

    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types, secondary=available_model_types,
        timeout=cur_lead_time)

    # Create the genetic programming-based composer, that allow to find
    # the optimal structure of the composite model
    builder = ComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    # run the search of best suitable model
    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)
    pipeline_evo_composed.fit(input_data=dataset_to_compose)

    return pipeline_evo_composed


def apply_model_to_data(model: Pipeline, data_path: str):
    df, file_path = create_multi_clf_examples_from_excel(data_path, return_df=True)
    dataset_to_apply = InputData.from_csv(file_path, target_columns=None)
    evo_predicted = model.predict(dataset_to_apply)
    df['forecast'] = probs_to_labels(evo_predicted.predict)
    return df


def validate_model_quality(model: Pipeline, data_path: str):
    dataset_to_validate = InputData.from_csv(data_path)
    predicted_labels = model.predict(dataset_to_validate).predict

    roc_auc_valid = round(roc_auc(y_true=test_data.target,
                                  y_score=predicted_labels,
                                  multi_class='ovo',
                                  average='macro'), 3)
    return roc_auc_valid


if __name__ == '__main__':
    file_path_first = r'../../data/example1.xlsx'
    file_path_second = r'../../data/example2.xlsx'
    file_path_third = r'../../data/example3.xlsx'

    train_file_path, test_file_path = create_multi_clf_examples_from_excel(file_path_first)
    test_data = InputData.from_csv(test_file_path)

    fitted_model = get_model(train_file_path)

    fitted_model.show()

    roc_auc = validate_model_quality(fitted_model, test_file_path)
    print(f'ROC AUC metric is {roc_auc}')

    final_prediction_first = apply_model_to_data(fitted_model, file_path_second)
    print(final_prediction_first['forecast'])

    final_prediction_second = apply_model_to_data(fitted_model, file_path_third)
    print(final_prediction_second['forecast'])
