from functools import partial, wraps
from itertools import chain
from random import choice, randint, random, sample
from typing import TYPE_CHECKING, Optional, Dict, Callable

from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_root_level, distance_to_primary_level, graph_has_cycle
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.operators.base_mutations import \
    add_as_child, add_separate_parent_node, add_intermediate_node, single_edge_mutation, single_add_mutation, \
    single_change_mutation, single_drop_mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.utilities.data_structures import ComparableEnum as Enum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


def _extract_pipelines(pipeline: Pipeline) -> Dict[str, Pipeline]:
    """ Get all pipelines from pipeline with atomized nodes as plane list
        Return dict with key as node uid (where pipeline is stored in atomized models)
        and values as pipelines """
    pipelines = {'': pipeline}
    for node in pipeline.nodes:
        if isinstance(node.operation, AtomizedModel):
            extracted_pipelines = _extract_pipelines(node.operation.pipeline)
            for k, v in extracted_pipelines.items():
                pipelines[k or node.uid] = v
    return pipelines


def _insert_pipelines(full_pipeline: Pipeline, node_uid: str, pipeline: Pipeline) -> Pipeline:
    """ Insert pipeline to full_pipeline with atomized model in node with uid node_uid """
    if node_uid == '':
        full_pipeline = pipeline
    else:
        # look for node with uid == node_uid
        nodes = full_pipeline.nodes[:]
        while nodes:
            node = nodes.pop()
            if node.uid == node_uid:
                break
            if isinstance(node.operation, AtomizedModel):
                nodes.extend(node.operation.pipeline.nodes)
        else:
            raise ValueError(f"Unknown node uid: {node_uid}")
        if not isinstance(node.operation, AtomizedModel):
            raise ValueError(f"Cannot insert pipeline to non AtomizedModel")
        node.operation.pipeline = pipeline
    return full_pipeline


MutationFun = Callable[[Pipeline, GraphRequirements, GraphGenerationParams, GPAlgorithmParameters], Pipeline]


def atomized_mutation(mutation_fun: MutationFun) -> MutationFun:
    # @wraps
    def mutation_for_atomized_graph(pipeline: Pipeline,
                                    requirements: GraphRequirements,
                                    graph_gen_params: GraphGenerationParams,
                                    parameters: GPAlgorithmParameters,
                                    ) -> OptGraph:
        # get all pipelines
        pipelines = _extract_pipelines(pipeline)

        # select pipeline to mutate
        node_uid, pipeline_to_mutate = choice(list(pipelines.items()))

        # mutate with GOLEM mutation fun
        mutated_pipeline = mutation_fun(graph=pipeline_to_mutate,
                                        requirements=requirements,
                                        graph_gen_params=graph_gen_params,
                                        parameters=parameters)

        # insert mutated pipeline inside origin pipeline
        new_pipeline = _insert_pipelines(pipeline, node_uid, mutated_pipeline)

        return new_pipeline

    return mutation_for_atomized_graph


fedot_single_edge_mutation = atomized_mutation(single_edge_mutation)
fedot_single_add_mutation = atomized_mutation(single_add_mutation)
fedot_single_change_mutation = atomized_mutation(single_change_mutation)
fedot_single_drop_mutation = atomized_mutation(single_drop_mutation)
