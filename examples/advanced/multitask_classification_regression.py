import os
from datetime import timedelta

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from test.unit.api.test_api_cli_params import project_root_path


def get_multitask_pipeline():
    logit_node = PrimaryNode('logit')
    data_source_node = PrimaryNode('data_source_table/regression')
    final_node = SecondaryNode('dtreg', nodes_from=[logit_node, data_source_node])
    return Pipeline(final_node)


def prepare_multitask_data() -> (MultiModalData, MultiModalData):
    """ Load data for multitask regression / classification pipeline """
    ex_data = os.path.join(project_root_path, 'examples/data')
    train_df = pd.read_csv(os.path.join(ex_data, 'train_synthetic_regression_classification.csv'))
    test_df = pd.read_csv(os.path.join(ex_data, 'test_synthetic_regression_classification.csv'))

    # Data for classification
    class_task = Task(TaskTypesEnum.classification)
    class_train = InputData(idx=np.arange(0, len(train_df)), features=np.array(train_df[['feature_1', 'feature_2']]),
                            target=np.array(train_df['class']), task=class_task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(is_main_target=False))
    class_test = InputData(idx=np.arange(0, len(test_df)), features=np.array(test_df[['feature_1', 'feature_2']]),
                           target=None, task=class_task, data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData(is_main_target=False))

    # Data for regression
    task = Task(TaskTypesEnum.regression)
    reg_train = InputData(idx=np.arange(0, len(train_df)), features=np.array(train_df[['feature_1', 'feature_2']]),
                          target=np.array(train_df['concentration']), task=task, data_type=DataTypesEnum.table)
    reg_test = InputData(idx=np.arange(0, len(test_df)), features=np.array(test_df[['feature_1', 'feature_2']]),
                         target=None, task=task, data_type=DataTypesEnum.table)

    train_multimodal = MultiModalData({'logit': class_train,
                                       'data_source_table/regression': reg_train})

    test_multimodal = MultiModalData({'logit': class_test,
                                      'data_source_table/regression': reg_test})
    return train_multimodal, test_multimodal


def launch_multitask_example(with_tuning: bool = False):
    """
    Demonstration of an example with running a multitask pipeline.
    Synthetic data is used. Task: predict the category of the substance (column "class") <- classification,
    and then predict the concentration based on the predicted category (column "concentration") <- regression.

    :param with_tuning: is tuning required or not
    """
    train_input, test_input = prepare_multitask_data()
    multitask_pipeline = get_multitask_pipeline()

    if with_tuning:
        tuner = TunerBuilder(train_input.task)\
            .with_tuner(PipelineTuner)\
            .with_metric(RegressionMetricsEnum.MAE)\
            .with_iterations(100)\
            .with_timeout(timedelta(minutes=2))\
            .build(train_input)
        multitask_pipeline = tuner.tune(multitask_pipeline)

    multitask_pipeline.fit(train_input)
    side_pipeline = multitask_pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.classification)

    # Replace the name of main "data source" in preprocessor
    side_pipeline.preprocessor.main_target_source_name = 'logit'
    side_output = side_pipeline.predict(test_input, output_mode='labels')
    output = multitask_pipeline.predict(test_input)

    print(f'Predicted classes: {np.ravel(side_output.predict)}')
    print(f'Predicted concentrations: {np.ravel(output.predict)}')


if __name__ == '__main__':
    launch_multitask_example(with_tuning=True)
