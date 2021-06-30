from copy import deepcopy
from functools import partial
from random import choice, randint, random
from typing import Any, Callable, TYPE_CHECKING, Union

from fedot.core.composer.constraint import constraint_function
from fedot.core.dag.graph import List
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.utils import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_optimiser import GraphGenerationParams

MAX_NUM_OF_ATTEMPTS = 100


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    none = 'none'


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


def get_mutation_prob(mut_id, node):
    """ Function returns mutation probability for certain node in the graph

    :param mut_id: MutationStrengthEnum mean weak or strong mutation
    :param node: root node of the graph
    :return mutation_prob: mutation probability
    """

    default_mutation_prob = 0.7
    if mut_id in list(MutationStrengthEnum):
        mutation_strength = mut_id.value
        mutation_prob = mutation_strength / (node.distance_to_primary_level + 1)
    else:
        mutation_prob = default_mutation_prob
    return mutation_prob


def will_mutation_be_applied(mutation_prob, mutation_type) -> bool:
    return not (random() > mutation_prob or mutation_type == MutationTypesEnum.none)


def mutation(types: List[Union[MutationTypesEnum, Callable]], params: 'GraphGenerationParams',
             ind: Individual, requirements, log: Log,
             max_depth: int = None, add_to_history=True) -> Any:
    """ Function apply mutation operator to graph """
    max_depth = max_depth if max_depth else requirements.max_depth
    mutation_prob = requirements.mutation_prob
    mutation_type = choice(types)
    is_custom_mutation = isinstance(mutation_type, Callable)
    if will_mutation_be_applied(mutation_prob, mutation_type):
        if mutation_type in mutation_by_type or is_custom_mutation:
            for _ in range(MAX_NUM_OF_ATTEMPTS):
                if is_custom_mutation:
                    mutation_func = mutation_type
                else:
                    mutation_func = mutation_by_type[mutation_type]

                input_obj = deepcopy(ind.graph)
                is_custom_operator = isinstance(input_obj, OptGraph)
                if is_custom_operator:
                    input_obj = params.adapter.restore(input_obj)

                new_graph = mutation_func(input_obj, requirements=requirements,
                                          params=params,
                                          max_depth=max_depth)

                if is_custom_operator:
                    new_graph = params.adapter.adapt(new_graph)

                is_correct_graph = constraint_function(new_graph,
                                                       params)
                if is_correct_graph:
                    new_individual = Individual(new_graph)
                    if add_to_history:
                        new_individual.parent_operators.append(
                            ParentOperator(operator_type='mutation',
                                           operator_name=str(mutation_type),
                                           parent_objects=[params.adapter.restore_as_template(ind.graph)]))
                    return new_individual

        elif mutation_type != MutationTypesEnum.none:
            raise ValueError(f'Required mutation type is not found: {mutation_type}')
        log.debug('Number of mutation attempts exceeded. '
                  'Please check composer requirements for correctness.')
    return deepcopy(ind)


def simple_mutation(graph: Any, requirements, **kwargs) -> Any:
    """
    This type of mutation is passed over all nodes of the tree started from the root node and changes
    nodes’ operations with probability - 'node mutation probability'
    which is initialised inside the function
    """

    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  node=graph.root_node)

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probability:
                secondary_node = OptNode(content=choice(requirements.secondary),
                                         nodes_from=node.nodes_from)
                graph.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probability:
                primary_node = OptNode(content=choice(requirements.primary))
                graph.update_node(node, primary_node)

    replace_node_to_random_recursive(graph.root_node)

    return graph


def _single_add_mutation(pipeline: Any, requirements, pipeline_generation_params):
    """
    Add new node between two sequential existing modes
    """
    node = choice(pipeline.nodes)
    if node.nodes_from:
        new_node = OptNode(content=choice(requirements.secondary))
        pipeline.operator.actualise_old_node_children(node, new_node)
        new_node.nodes_from = [node]
        pipeline.nodes.append(new_node)
    return pipeline


def _tree_growth(pipeline: Any, requirements, params, max_depth: int, local_growth=True):
    """
    This mutation selects a random node in a tree, generates new subtree,
    and replaces the selected node's subtree.
    """
    random_layer_in_pipeline = randint(0, pipeline.depth - 1)
    node_from_pipeline = choice(pipeline.operator.nodes_from_layer(random_layer_in_pipeline))
    if local_growth:
        is_primary_node_selected = (not node_from_pipeline.nodes_from) or (
                node_from_pipeline.nodes_from and
                node_from_pipeline != pipeline.root_node
                and randint(0, 1))
    else:
        is_primary_node_selected = \
            randint(0, 1) and \
            not pipeline.operator.distance_to_root_level(node_from_pipeline) < max_depth
    if is_primary_node_selected:
        new_subtree = OptNode(content=choice(requirements.primary))
    else:
        if local_growth:
            max_depth = node_from_pipeline.distance_to_primary_level
        else:
            max_depth = max_depth - pipeline.operator.distance_to_root_level(node_from_pipeline)
        new_subtree = random_graph(params=params, requirements=requirements,
                                   max_depth=max_depth).root_node
    pipeline.update_subtree(node_from_pipeline, new_subtree)
    return pipeline


def growth_mutation(pipeline: Any, requirements, params, max_depth: int, local_growth=True) -> Any:
    """
    This mutation adds new nodes to the graph (just single node between existing nodes or new subtree).
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """

    if random() > 0.5:
        # simple growth (one node can be added)
        return _single_add_mutation(pipeline, requirements, params)
    else:
        # advanced growth (several nodes can be added)
        return _tree_growth(pipeline, requirements, params, max_depth, local_growth)


def reduce_mutation(graph: Any, requirements, **kwargs) -> Any:
    """
    Selects a random node in a tree, then removes its subtree. If the current arity of the node's
    parent is more than the specified minimal arity, then the selected node is also removed.
    Otherwise, it is replaced by a random primary node.
    """

    nodes = [node for node in graph.nodes if node is not graph.root_node]
    node_to_del = choice(nodes)
    children = graph.operator.node_children(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in children])
    if is_possible_to_delete:
        graph.delete_subtree(node_to_del)
    else:
        primary_node = OptNode(content=choice(requirements.primary))
        graph.update_subtree(node_to_del, primary_node)
    return graph


# TODO move to composer


mutation_by_type = {
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.reduce: reduce_mutation,
}
