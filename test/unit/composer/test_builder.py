import datetime

from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


def test_gp_composer_builder():
    task = Task(TaskTypesEnum.classification)

    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=4,
        crossover_prob=0.8, mutation_prob=1, max_lead_time=datetime.timedelta(minutes=5))

    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

    builder_with_custom_params = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_optimiser_parameters(optimiser_parameters)

    composer_with_custom_params = builder_with_custom_params.build()

    assert composer_with_custom_params.optimiser.parameters.genetic_scheme_type == scheme_type
    assert composer_with_custom_params.metrics == metric_function
    assert composer_with_custom_params.composer_requirements.pop_size == 5
    assert composer_with_custom_params.composer_requirements.mutation_prob == 1

    builder_with_default_params = GPComposerBuilder(task=task)
    composer_with_default_params = builder_with_default_params.build()

    default_metric = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC.ROCAUC_penalty)

    assert composer_with_default_params.optimiser.parameters.genetic_scheme_type == GeneticSchemeTypesEnum.generational
    assert composer_with_default_params.metrics == default_metric
    assert composer_with_default_params.composer_requirements.pop_size == 20
    assert composer_with_default_params.composer_requirements.mutation_prob == 0.8
