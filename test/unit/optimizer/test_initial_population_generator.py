from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.pipelines.verification import rules_by_task
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.optimisers.initial_graphs_generator import InitialPopulationGenerator


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
