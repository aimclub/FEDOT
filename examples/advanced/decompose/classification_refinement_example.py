import random

import numpy as np

from cases.credit_scoring.credit_scoring_problem import get_scoring_data, calculate_validation_metric
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(1)
np.random.seed(1)


def get_refinement_pipeline():
    """ Create 3-level pipeline with class_decompose node """
    node_scaling = PrimaryNode('scaling')
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('class_decompose', nodes_from=[node_logit, node_scaling])
    node_rfr = SecondaryNode('rfr', nodes_from=[node_decompose])
    node_rf = SecondaryNode('rf', nodes_from=[node_rfr, node_logit])

    pipeline = Pipeline(node_rf)
    return pipeline


def get_non_refinement_pipeline():
    """ Create 3-level pipeline without class_decompose node """
    node_scaling = PrimaryNode('scaling')
    node_rf = SecondaryNode('rf', nodes_from=[node_scaling])
    node_logit = SecondaryNode('logit', nodes_from=[node_scaling])
    node_rf = SecondaryNode('rf', nodes_from=[node_logit, node_rf])
    pipeline = Pipeline(node_rf)
    return pipeline


def display_roc_auc(pipeline_to_validate, test_dataset, pipeline_name: str):
    roc_auc_metric = calculate_validation_metric(pipeline_to_validate, test_dataset)
    print(f'{pipeline_name} ROC AUC: {roc_auc_metric:.4f}')


def run_refinement_scoring_example(train_path, test_path, with_tuning=False):
    """ Function launch example with error modeling for classification task

    :param train_path: path to the csv file with training sample
    :param test_path: path to the csv file with test sample
    :param with_tuning: is it need to tune pipelines or not
    """

    task = Task(TaskTypesEnum.classification)
    train_dataset = InputData.from_csv(train_path, task=task)
    test_dataset = InputData.from_csv(test_path, task=task)

    # Get and fit pipelines
    no_decompose_c = get_non_refinement_pipeline()
    decompose_c = get_refinement_pipeline()

    no_decompose_c.fit(train_dataset)
    decompose_c.fit(train_dataset)

    # Check metrics for both pipelines
    display_roc_auc(no_decompose_c, test_dataset, 'Non decomposition pipeline')
    display_roc_auc(decompose_c, test_dataset, 'With decomposition pipeline')

    if with_tuning:
        tuner = TunerBuilder(task) \
            .with_tuner(PipelineTuner)\
            .with_metric(ClassificationMetricsEnum.ROCAUC)\
            .with_iterations(30) \
            .build(train_dataset)
        no_decompose_c = tuner.tune(no_decompose_c)
        decompose_c = tuner.tune(decompose_c)

        no_decompose_c.fit(test_dataset)
        decompose_c.fit(test_dataset)

        display_roc_auc(no_decompose_c, test_dataset, 'Non decomposition pipeline after tuning')
        display_roc_auc(decompose_c, test_dataset, 'With decomposition pipeline after tuning')


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_refinement_scoring_example(full_path_train, full_path_test, with_tuning=True)
