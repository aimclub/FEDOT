from functools import reduce
from random import choice

import pytest

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.initial_graphs_generator import InitialPopulationGenerator
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third


@pytest.fixture(scope='module')
def setup_test():
    task = Task(TaskTypesEnum.classification)
    available_model_types = get_operations_for_task(task=task, mode='model')
    requirements = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types)
    graph_generation_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                                    advisor=PipelineChangeAdvisor(task),
                                                    rules_for_constraint=rules_by_task(task.task_type))
    return requirements, graph_generation_params, InitialPopulationGenerator(graph_generation_params, requirements)


def test_random_initial_population(setup_test):
    requirements, graph_generation_params, initial_population_generator = setup_test
    initial_population_generator.with_initial_graphs(()).with_custom_generation_function(None)
    generated_population = initial_population_generator.get_initial_graphs(pop_size=3)
    max_depth = requirements.max_depth
    verifier = graph_generation_params.verifier
    print([graph.depth for graph in generated_population], max_depth)
    assert len(generated_population) == 3, \
        "If no initial graphs provided InitialPopulationGenerator returns randomly generated graphs."
    assert all(graph.depth <= max_depth for graph in generated_population)
    assert all(verifier(graph) for graph in generated_population)


def test_initial_graphs_as_initial_population(setup_test):
    requirements, graph_generation_params, initial_population_generator = setup_test
    initial_graphs = [pipeline_first(), pipeline_second(), pipeline_third()]
    initial_population_generator.with_initial_graphs(initial_graphs)

    generated_population = initial_population_generator.get_initial_graphs(pop_size=3)
    assert generated_population == initial_graphs

    generated_population1 = initial_population_generator.get_initial_graphs(pop_size=4)
    assert generated_population1 == initial_graphs

    generated_population = initial_population_generator.get_initial_graphs(pop_size=2)
    assert len(generated_population) == 2
    assert all(graph in initial_graphs for graph in generated_population)


@pytest.mark.parametrize('pop_size', [3, 4])
def test_initial_population_generation_function(setup_test, pop_size):
    requirements, graph_generation_params, initial_population_generator = setup_test
    initial_population_generator.with_custom_generation_function(
        lambda: choice([pipeline_first(), pipeline_second(), pipeline_third()]))
    verifier = graph_generation_params.verifier

    generated_population = initial_population_generator.get_initial_graphs(pop_size=pop_size)
    assert len(generated_population) <= 3
    assert all(verifier(graph) for graph in generated_population)
    unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, generated_population, [])
    assert len(unique) == len(generated_population)
