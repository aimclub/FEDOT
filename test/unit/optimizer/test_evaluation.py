import os
from functools import partial

import pytest

from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher, SimpleDispatcher
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.objective import Objective
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.api.test_api_cli_params import project_root_path
from test.unit.models.test_model import classification_dataset



def first_pipeline():
    """
    Returns pipeline with structure
    knn -> prediction
    """
    pipeline_builder = PipelineBuilder().add_node('knn')
    pipeline = pipeline_builder.to_pipeline()
    return pipeline


def second_pipeline():
    """
    Returns pipeline with structure
      knn
         \
           logit -> prediction
         /
      svc
    """
    pipeline_builder = PipelineBuilder()\
        .add_sequence('knn', branch_idx=0)\
        .add_sequence('svc', branch_idx=1)\
        .join_branches('logit')
    pipeline = pipeline_builder.to_pipeline()
    return pipeline


def third_pipeline():
    """
    Returns pipeline with structure
    rf -> logit -> prediction
    """
    pipeline_builder = PipelineBuilder()\
        .add_node('rf')\
        .add_node('logit')
    pipeline = pipeline_builder.to_pipeline()
    return pipeline


def fourth_pipeline():
    """
    Returns pipeline with structure
    rf -> logit
               \
                svc -> rf -> prediction
               |
            scv
    """
    pipeline_builder = PipelineBuilder()\
        .add_sequence('rf', 'logit', branch_idx=0)\
        .add_sequence('svc', branch_idx=1)\
        .join_branches('svc')
    pipeline = pipeline_builder.to_pipeline()
    return pipeline


def classification_random_forest_pipeline():
    """
    Returns pipeline with the following structure:

    scaling -> rf -> final prediction
    """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('rf', nodes_from=[node_scaling])
    return Pipeline(node_final)


def sample_pipeline():
    return Pipeline(SecondaryNode(operation_type='logit',
                                  nodes_from=[PrimaryNode(operation_type='rf'),
                                              PrimaryNode(operation_type='scaling')]))


def get_classification_data():
    file_path = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    input_data = InputData.from_csv(file_path, task=Task(TaskTypesEnum.classification))
    return input_data


def test_multiprocessingdispatcher_without_timelimit_without_multiprocessing(classification_dataset):
    metric = ClassificationMetricsEnum.accuracy
    objective = Objective(metric)
    # input_data = get_classification_data()
    input_data = classification_dataset

    objective_function_with_data = partial(objective, reference_data=input_data)
    adapter = PipelineAdapter()
    evaluator = SimpleDispatcher(adapter).dispatch(objective_function_with_data)

    # pipelines = [first_pipeline(), second_pipeline(), third_pipeline(), fourth_pipeline()]
    pipelines = [sample_pipeline()]
    list(map(lambda x: x.fit(input_data=input_data), pipelines))
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]
    evaluator(population)
