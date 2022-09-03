import datetime
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.credit_scoring.credit_scoring_problem import get_scoring_data
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_optimizer import GPGraphOptimizerParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.visualisation.opt_viz_extra import OptHistoryExtraVisualizer

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
                               is_visualise=False):
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
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, timeout=timeout,
        start_depth=2)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.parameter_free
    optimiser_parameters = GPGraphOptimizerParameters(genetic_scheme_type=scheme_type,
                                                      selection_types=[SelectionTypesEnum.spea2])

    # Create builder for composer and set composer params
    builder = ComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metrics).with_optimiser_params(parameters=optimiser_parameters)

    # Create GP-based composer
    composer = builder.build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipelines_evo_composed = composer.compose_pipeline(data=dataset_to_compose)

    composer.history.to_csv()

    if is_visualise:
        results_visualization(composed_pipelines=pipelines_evo_composed, history=composer.history)

    pipelines_roc_auc = []
    for pipeline_num, pipeline_evo_composed in enumerate(pipelines_evo_composed):

        pipeline_evo_composed.fine_tune_primary_nodes(input_data=dataset_to_compose,
                                                      iterations=50)

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
    run_credit_scoring_problem(full_path_train, full_path_test, is_visualise=True)
