from golem.core.optimisers.genetic.gp_params import GPGraphOptimizerParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from test.unit.optimizer.gp_operators.test_gp_operators import generate_pipeline_with_single_node
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second


def test_crossover():
    adapter = PipelineAdapter()
    graph_example_first = adapter.adapt(pipeline_first())
    graph_example_second = adapter.adapt(pipeline_second())

    requirements = PipelineComposerRequirements()
    opt_parameters = GPGraphOptimizerParameters(crossover_types=[CrossoverTypesEnum.none], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, get_pipeline_generation_params())
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second

    opt_parameters = GPGraphOptimizerParameters(crossover_types=[CrossoverTypesEnum.subtree], crossover_prob=0)
    crossover = Crossover(opt_parameters, requirements, get_pipeline_generation_params())
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second


def test_crossover_with_single_node():
    adapter = PipelineAdapter()
    graph_example_first = adapter.adapt(generate_pipeline_with_single_node())
    graph_example_second = adapter.adapt(generate_pipeline_with_single_node())

    requirements = PipelineComposerRequirements()
    graph_params = get_pipeline_generation_params(requirements)

    for crossover_type in CrossoverTypesEnum:
        opt_parameters = GPGraphOptimizerParameters(crossover_types=[crossover_type], crossover_prob=1)
        crossover = Crossover(opt_parameters, requirements, graph_params)
        new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])

        assert new_graphs[0].graph == graph_example_first
        assert new_graphs[1].graph == graph_example_second
