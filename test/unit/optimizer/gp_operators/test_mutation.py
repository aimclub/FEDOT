from copy import deepcopy
from itertools import chain
from pathlib import Path

import pytest
from typing import Any, List, Optional, Type, Callable

from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationStrengthEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.composer.gp_composer.specific_operators import boosting_mutation
from fedot.core.data.data import InputData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.optimisers.genetic_operators.mutation import fedot_single_edge_mutation
from test.integration.composer.test_composer import to_categorical_codes
from test.unit.dag.test_graph_utils import find_first
from test.unit.tasks.test_forecasting import get_ts_data


def get_requirements_and_params_for_task(task: TaskTypesEnum):
    ops = get_operations_for_task(Task(task))
    req = PipelineComposerRequirements(primary=ops, secondary=ops, max_depth=2)
    gen_params = get_pipeline_generation_params(requirements=req, task=Task(task))
    return req, gen_params


def file_data():
    test_file_path = Path(__file__).parents[3].joinpath('data', 'simple_classification.csv')
    input_data = InputData.from_csv(test_file_path)
    input_data.idx = to_categorical_codes(categorical_ids=input_data.idx)
    return input_data


def get_mutation_obj(mutation_types: Optional[List[Any]] = None) -> Mutation:
    """
    Function for initializing mutation interface
    """
    task = Task(TaskTypesEnum.classification)
    operations = ['logit', 'normalization']

    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)

    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES,
                                                  task=task)
    kwargs = dict(mutation_strength=MutationStrengthEnum.strong,
                  mutation_prob=1)
    if mutation_types is not None:
        kwargs = {'mutation_types': mutation_types, **kwargs}
    parameters = GPAlgorithmParameters(**kwargs)

    mutation = Mutation(parameters, requirements, graph_params)
    return mutation


def get_simple_linear_graph() -> OptGraph:
    """
    Returns simple linear graph
    """
    pipeline = PipelineBuilder().add_node('scaling').add_node('poly_features').add_node('rf').build()
    return PipelineAdapter().adapt(pipeline)


def get_simple_linear_boosting_pipeline() -> Pipeline:
    """
    Returns simple linear graph with boosting
    """
    node_scaling = PipelineNode('scaling')
    node_pf = PipelineNode('poly_features', nodes_from=[node_scaling])
    node_rf = PipelineNode('rf', nodes_from=[node_pf])
    node_decompose = PipelineNode('class_decompose', nodes_from=[node_pf, node_rf])
    node_linear = PipelineNode('linear', nodes_from=[node_decompose])
    final_node = PipelineNode('logit', nodes_from=[node_linear, node_rf])
    pipeline = Pipeline(final_node)
    return pipeline


def get_ts_forecasting_graph() -> OptGraph:
    """
    Returns simple linear graph for ts forecasting
    """
    pipeline = PipelineBuilder().add_node('smoothing').add_node('ar').build()

    return PipelineAdapter().adapt(pipeline)


def get_ts_forecasting_graph_with_boosting() -> Pipeline:
    """
    Returns simple linear graph for ts forecasting with boosting
    """
    node_init = PipelineNode('smoothing')
    node_model = PipelineNode('ar', nodes_from=[node_init])
    node_lagged = PipelineNode('lagged', nodes_from=[node_init])
    node_decompose = PipelineNode('decompose',
                                  [node_model, node_lagged])
    node_ridge = PipelineNode('ridge', nodes_from=[node_decompose])
    node_final = PipelineNode('ridge', nodes_from=[node_ridge, node_model])
    pipeline = Pipeline(node_final)
    return pipeline


def get_graph_with_two_nested_atomized_models(atomized_model):
    simple_pipeline = (PipelineBuilder()
                       .add_node('scaling')
                       .add_branch('linear', 'poly_features')
                       .grow_branches('rf', 'catboost')
                       .join_branches('ridge')
                       .build())

    node1 = PipelineNode('a')
    node2 = PipelineNode('b', nodes_from=[node1])
    node3 = PipelineNode(atomized_model(simple_pipeline), nodes_from=[node1])
    node4 = PipelineNode('c', nodes_from=[node1, node3])
    node5 = PipelineNode('d', nodes_from=[node2, node4])
    node6 = PipelineNode('e', nodes_from=[node2, node5])
    pipeline_with_atomized = Pipeline(node6)

    node1 = PipelineNode('1')
    node2 = PipelineNode('2', nodes_from=[node1])
    node3 = PipelineNode(atomized_model(pipeline_with_atomized), nodes_from=[node1])
    node4 = PipelineNode('3', nodes_from=[node1, node3])
    node5 = PipelineNode('4', nodes_from=[node2, node4])
    node6 = PipelineNode('5', nodes_from=[node2, node5])
    pipeline_with_atomized = Pipeline(node6)
    return PipelineAdapter().adapt(pipeline_with_atomized)


def test_boosting_mutation_for_linear_graph():
    """
    Tests boosting mutation can add correct boosting cascade
    """

    graph = PipelineAdapter().restore(get_simple_linear_graph())
    boosting_graph = get_simple_linear_boosting_pipeline()
    requirements = PipelineComposerRequirements(primary=['logit'],
                                                secondary=['logit'])
    pipeline = boosting_mutation(graph,
                                 requirements,
                                 get_pipeline_generation_params(requirements=requirements,
                                                                rules_for_constraint=DEFAULT_DAG_RULES,
                                                                task=Task(TaskTypesEnum.classification)))
    data = file_data()
    pipeline.fit(data)
    result = pipeline.predict(data)
    assert pipeline.descriptive_id == boosting_graph.descriptive_id
    assert result is not None


def test_boosting_mutation_for_non_lagged_ts_model():
    """
    Tests boosting mutation can add correct boosting cascade for ts forecasting with non-lagged model
    """

    graph = PipelineAdapter().restore(get_ts_forecasting_graph())

    boosting_graph = get_ts_forecasting_graph_with_boosting()
    requirements = PipelineComposerRequirements(primary=['ridge'],
                                                secondary=['ridge'])
    pipeline = boosting_mutation(graph,
                                 requirements,
                                 get_pipeline_generation_params(requirements=requirements,
                                                                rules_for_constraint=DEFAULT_DAG_RULES,
                                                                task=Task(TaskTypesEnum.ts_forecasting)))
    data_train, data_test = get_ts_data()
    pipeline.fit(data_train)
    result = pipeline.predict(data_test)
    assert boosting_graph.descriptive_id == pipeline.descriptive_id
    assert result is not None


@pytest.mark.parametrize('pipeline, requirements, params',
                         [(PipelineBuilder().add_node('scaling').add_node('rf').build(),
                           *get_requirements_and_params_for_task(TaskTypesEnum.classification)),
                          (PipelineBuilder().add_node('smoothing').add_node('ar').build(),
                           *get_requirements_and_params_for_task(TaskTypesEnum.ts_forecasting))
                          ])
def test_boosting_mutation_changes_pipeline(pipeline: Pipeline, requirements: PipelineComposerRequirements,
                                            params: GraphGenerationParams):
    new_pipeline = deepcopy(pipeline)
    new_pipeline = boosting_mutation(new_pipeline, requirements, params)
    assert new_pipeline.descriptive_id != pipeline.descriptive_id
    assert 'class_decompose' in new_pipeline.descriptive_id or 'decompose' in new_pipeline.descriptive_id


def test_no_opt_or_graph_nodes_after_mutation():
    adapter = PipelineAdapter()
    graph = get_simple_linear_graph()
    mutation = get_mutation_obj()
    for mut in mutation.parameters.mutation_types:
        graph, _ = mutation._adapt_and_apply_mutation(new_graph=graph, mutation_type=mut)
    new_pipeline = adapter.restore(graph)

    assert not find_first(new_pipeline, lambda n: type(n) in (GraphNode, OptNode))


@pytest.mark.parametrize('atomized_model',
                         (AtomizedModel, ))
@pytest.mark.parametrize('mutation_type',
                         (fedot_single_edge_mutation, ))
def test_fedot_mutation_with_atomized_models(atomized_model: Type[AtomizedModel],
                                             mutation_type: Callable[[OptGraph], OptGraph]):

    def extract_all_graphs(graph: OptGraph):
        """ get all graphs from graph with atomized nodes as plane list"""
        atomized_nodes = [node for node in graph.nodes if 'atomized' in node.name.lower()]
        atomized_graphs = list(chain(*[extract_all_graphs(node.content['inner_graph']) for node in atomized_nodes]))
        return [graph] + atomized_graphs

    mutation = get_mutation_obj(mutation_types=[mutation_type])
    # check that mutation_type has been set correctly
    assert len(mutation.parameters.mutation_types) == 1
    assert mutation.parameters.mutation_types[0] is mutation_type

    # make mutation some times
    mut = mutation.parameters.mutation_types[0]
    origin_graphs = extract_all_graphs(get_graph_with_two_nested_atomized_models(atomized_model))
    all_mutations = [0, 0, 0]
    for _ in range(20):
        graph, _ = mutation._adapt_and_apply_mutation(new_graph=deepcopy(origin_graphs[0]), mutation_type=mut)
        graphs = extract_all_graphs(graph)

        # check that there was the only one mutation in all graph
        assert sum(x != y for x, y in zip(origin_graphs, graphs)) == 1

        all_mutations = [x + (y != z) for x, y, z in zip(all_mutations, origin_graphs, graphs)]

    # check that all graphs receive at least 20% of mutations share
    assert all(x / sum(all_mutations) > 0.2 for x in all_mutations)

