from functools import reduce
from random import choice

import pytest
from golem.core.optimisers.genetic.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.initial_graphs_generator import InitialPopulationGenerator

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third


def setup_test(pop_size):
    task = Task(TaskTypesEnum.classification)
    available_model_types = get_operations_for_task(task=task, mode='model')
    requirements = PipelineComposerRequirements(primary=available_model_types,
                                                secondary=available_model_types)
    graph_generation_params = get_pipeline_generation_params(requirements=requirements,
                                                             rules_for_constraint=rules_by_task(task.task_type),
                                                             task=task)
    generator = InitialPopulationGenerator(pop_size, graph_generation_params, requirements)
    return requirements, graph_generation_params, generator


def test_random_initial_population():
    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=3)
    generated_population = initial_population_generator()
    max_depth = requirements.max_depth
    verifier = graph_generation_params.verifier
    assert len(generated_population) == 3, \
        "If no initial graphs provided InitialPopulationGenerator returns randomly generated graphs."
    assert all(graph.depth <= max_depth for graph in generated_population)
    assert all(verifier(graph) for graph in generated_population)


def test_initial_graphs_as_initial_population():
    adapter = PipelineAdapter()
    initial_graphs = adapter.adapt([pipeline_first(), pipeline_second(), pipeline_third()])

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=3)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert generated_population == initial_graphs

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=4)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert generated_population == initial_graphs

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=2)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert len(generated_population) == 2
    assert all(graph in initial_graphs for graph in generated_population)


@pytest.mark.parametrize('pop_size', [3, 4])
def test_initial_population_generation_function(pop_size):
    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=pop_size)
    initial_population_generator.with_custom_generation_function(
        lambda: choice([pipeline_first(), pipeline_second(), pipeline_third()]))
    verifier = graph_generation_params.verifier

    generated_population = initial_population_generator()
    assert len(generated_population) <= 3
    assert all(verifier(graph) for graph in generated_population)
    unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, generated_population, [])
    assert len(unique) == len(generated_population)
