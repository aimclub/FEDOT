import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def get_multimodal_pipeline():
    """ Generate multimodal pipeline for classification task with several tables """
    node_data_source_first = PipelineNode('data_source_table/first')
    node_data_source_second = PipelineNode('data_source_table/second')

    node_scaling_first = PipelineNode('scaling', nodes_from=[node_data_source_first])
    node_imputation_second = PipelineNode('simple_imputation', nodes_from=[node_data_source_second])

    node_final = PipelineNode('logit', nodes_from=[node_scaling_first, node_imputation_second])
    pipeline = Pipeline(node_final)

    return pipeline


def get_single_task_multimodal_tabular_data():
    """ Create MultiModalData object with two tables """
    task = Task(TaskTypesEnum.classification)

    # Create features table
    features_first = np.array([[0, 'a'], [1, 'a'], [2, 'b'], [3, np.nan], [4, 'a'],
                               [5, 'b'], [6, 'b'], [7, 'c'], [8, 'c']], dtype=object)
    features_second = np.array([[10, 'a'], [11, 'a'], [12, 'b'], [13, 'a'], [14, 'a'],
                                [15, 'b'], [16, 'b'], [17, 'c'], [18, 'c']], dtype=object)

    target = np.array(['true', 'false', 'true', 'false', 'false', 'false', 'false', 'true', 'true'], dtype=str)

    input_first = InputData(idx=np.arange(0, 9), features=features_first,
                            target=target, task=task, data_type=DataTypesEnum.table)
    input_second = InputData(idx=np.arange(0, 9), features=features_second,
                             target=target, task=task, data_type=DataTypesEnum.table)

    mm_data = MultiModalData({'data_source_table/first': input_first,
                              'data_source_table/second': input_second})

    pipeline = get_multimodal_pipeline()
    return mm_data, pipeline
