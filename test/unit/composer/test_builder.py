import datetime
from operator import eq

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.metrics import ROCAUC
from fedot.core.optimisers.gp_comp.gp_optimizer import GPGraphOptimizerParameters, GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def prepare_builder_with_custom_params(return_all: bool):
    task = Task(TaskTypesEnum.classification)

    available_model_types = OperationTypesRepository().suitable_operation(
        task_type=task.task_type)

    metric_function = ClassificationMetricsEnum.ROCAUC

    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=4,
        crossover_prob=0.8, mutation_prob=1,
        timeout=datetime.timedelta(minutes=5))

    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPGraphOptimizerParameters(
        genetic_scheme_type=scheme_type)

    builder_with_custom_params = ComposerBuilder(task=task).with_requirements(
        composer_requirements).with_metrics(
        metric_function).with_optimiser_params(parameters=optimiser_parameters)

    if return_all:
        return builder_with_custom_params, scheme_type, metric_function, task
    return builder_with_custom_params


def test_gp_composer_builder():
    builder, scheme_type, metric_function, task = prepare_builder_with_custom_params(return_all=True)
    default_complexity_metrics = builder._get_default_complexity_metrics()

    composer_with_custom_params = builder.build()

    assert composer_with_custom_params.optimizer.parameters.genetic_scheme_type == scheme_type
    assert composer_with_custom_params.composer_requirements.pop_size == 5
    assert composer_with_custom_params.composer_requirements.mutation_prob == 1
    assert all(map(eq, composer_with_custom_params.optimizer.objective.metrics,
                   [metric_function] + default_complexity_metrics))

    builder_with_default_params = ComposerBuilder(task=task)
    composer_with_default_params = builder_with_default_params.build()

    default_metric = ROCAUC.get_value_with_penalty

    assert composer_with_default_params.optimizer.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.generational
    assert composer_with_default_params.composer_requirements.pop_size == 20
    assert composer_with_default_params.composer_requirements.mutation_prob == 0.8
    assert all(map(eq, composer_with_default_params.optimizer.objective.metrics,
                   [default_metric] + default_complexity_metrics))
