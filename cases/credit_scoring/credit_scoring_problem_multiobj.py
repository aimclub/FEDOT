import datetime
import random

import numpy as np
from golem.core.optimisers.genetic.gp_params import GPGraphOptimizerParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.genetic.pipeline_composer_requirements import PipelineComposerRequirements
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer
from sklearn.metrics import roc_auc_score as roc_auc

from cases.credit_scoring.credit_scoring_problem import get_scoring_data
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

random.seed(12)
np.random.seed(12)


def results_visualization(history, composed_pipelines):
    visualiser = OptHistoryExtraVisualizer()
    visualiser.visualise_history(history)
    visualiser.pareto_gif_create(history.archive_history, history.individuals)
    visualiser.boxplots_gif_create(history.individuals)
    for pipeline_evo_composed in composed_pipelines:
        pipeline_evo_composed.show()


def calculate_validation_metric(pipeline: Pipeline, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = pipeline.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               timeout: datetime.timedelta = datetime.timedelta(minutes=5),
                               visualization=False):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='model')

    # the choice of the metric for the pipeline quality assessment during composition
    quality_metric = ClassificationMetricsEnum.ROCAUC
    complexity_metric = ComplexityMetricsEnum.node_num
    metrics = [quality_metric, complexity_metric]
    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
        timeout=timeout,
        num_of_generations=20
    )
    params = GPGraphOptimizerParameters(
        selection_types=[SelectionTypesEnum.spea2],
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
    )

    # Create composer and with required composer params
    composer = ComposerBuilder(task=task). \
        with_optimizer_params(params). \
        with_requirements(composer_requirements). \
        with_metrics(metrics). \
        build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipelines_evo_composed = composer.compose_pipeline(data=dataset_to_compose)

    composer.history.to_csv()

    if visualization:
        results_visualization(composed_pipelines=pipelines_evo_composed, history=composer.history)

    pipelines_roc_auc = []

    for pipeline_num, pipeline_evo_composed in enumerate(pipelines_evo_composed):

        tuner = TunerBuilder(task)\
            .with_tuner(SequentialTuner)\
            .with_iterations(50)\
            .with_metric(metrics[0])\
            .build(dataset_to_compose)
        nodes = pipeline_evo_composed.nodes
        for node_index, node in enumerate(nodes):
            if isinstance(node, PipelineNode) and node.is_primary:
                pipeline_evo_composed = tuner.tune_node(pipeline_evo_composed, node_index)

        pipeline_evo_composed.fit(input_data=dataset_to_compose)

        # the quality assessment for the obtained composite models
        roc_on_valid_evo_composed = calculate_validation_metric(pipeline_evo_composed,
                                                                dataset_to_validate)

        pipelines_roc_auc.append(roc_on_valid_evo_composed)
        if len(pipelines_evo_composed) > 1:
            print(f'Composed ROC AUC of pipeline {pipeline_num + 1} is {round(roc_on_valid_evo_composed, 3)}')

        else:
            print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return max(pipelines_roc_auc)


if __name__ == '__main__':
    full_path_train, full_path_test = get_scoring_data()
    run_credit_scoring_problem(full_path_train, full_path_test, visualization=True)
