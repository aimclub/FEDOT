from copy import deepcopy
from functools import partial
from random import choice, randint, random, sample
from typing import TYPE_CHECKING, Any, Callable, List, Union

import numpy as np

from fedot.core.composer.advisor import RemoveType
from fedot.core.composer.constraint import constraint_function
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.optimizer import GraphGenerationParams

MAX_NUM_OF_ATTEMPTS = 100
MAX_MUT_CYCLES = 5
STATIC_MUTATION_PROBABILITY = 0.7


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    single_add = 'single_add',
    single_change = 'single_change',
    single_drop = 'single_drop',
    single_edge = 'single_edge'

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


def _will_mutation_be_applied(mutation_prob, mutation_type) -> bool:
    return not (random() > mutation_prob or mutation_type == MutationTypesEnum.none)


def _adapt_and_apply_mutations(new_graph: Any, mutation_prob: float, types: List[Union[MutationTypesEnum, Callable]],
                               num_mut: int, requirements, params: 'GraphGenerationParams', max_depth: int):
    """
    Apply mutation in several iterations with specific adaptation of each graph
    """

    is_static_mutation_type = random() < STATIC_MUTATION_PROBABILITY
    static_mutation_type = choice(types)
    mutation_names = []
    for _ in range(num_mut):
        mutation_type = static_mutation_type \
            if is_static_mutation_type else choice(types)
        is_custom_mutation = isinstance(mutation_type, Callable)

        if is_custom_mutation:
            new_graph = params.adapter.restore(new_graph)
        else:
            if not isinstance(new_graph, OptGraph):
                new_graph = params.adapter.adapt(new_graph)
        new_graph = _apply_mutation(new_graph=new_graph, mutation_prob=mutation_prob,
                                    mutation_type=mutation_type, is_custom_mutation=is_custom_mutation,
                                    requirements=requirements, params=params, max_depth=max_depth)
        mutation_names.append(str(mutation_type))
        if not isinstance(new_graph, OptGraph):
            new_graph = params.adapter.adapt(new_graph)
        if is_custom_mutation:
            # custom mutation occurs once
            break
    return new_graph, mutation_names


def _apply_mutation(new_graph: Any, mutation_prob: float, mutation_type: Union[MutationTypesEnum, Callable],
                    is_custom_mutation: bool, requirements, params: 'GraphGenerationParams', max_depth: int):
    """
      Apply mutation for adapted graph
    """
    if _will_mutation_be_applied(mutation_prob, mutation_type):
        if mutation_type in mutation_by_type or is_custom_mutation:
            if is_custom_mutation:
                mutation_func = mutation_type
            else:
                mutation_func = mutation_by_type[mutation_type]
            new_graph = mutation_func(new_graph, requirements=requirements,
                                      params=params,
                                      max_depth=max_depth)
        elif mutation_type != MutationTypesEnum.none:
            raise ValueError(f'Required mutation type is not found: {mutation_type}')
    return new_graph


def mutation(types: List[Union[MutationTypesEnum, Callable]], params: 'GraphGenerationParams',
             ind: Individual, requirements, log: Log,
             max_depth: int = None) -> Any:
    """ Function apply mutation operator to graph """
    max_depth = max_depth if max_depth else requirements.max_depth
    mutation_prob = requirements.mutation_prob

    for _ in range(MAX_NUM_OF_ATTEMPTS):
        new_graph = deepcopy(ind.graph)
        num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)

        new_graph, mutation_names = _adapt_and_apply_mutations(new_graph=new_graph, mutation_prob=mutation_prob,
                                                               types=types, num_mut=num_mut,
                                                               requirements=requirements, params=params,
                                                               max_depth=max_depth)

        is_correct_graph = constraint_function(new_graph, params)
        if is_correct_graph:
            new_individual = Individual(new_graph)
            new_individual.parent_operators = deepcopy(ind.parent_operators)
            for mutation_name in mutation_names:
                new_individual.parent_operators.append(
                    ParentOperator(operator_type='mutation',
                                   operator_name=str(mutation_name),
                                   parent_individuals=[ind]))
            return new_individual

    log.debug('Number of mutation attempts exceeded. '
              'Please check composer requirements for correctness.')

    return deepcopy(ind)


def simple_mutation(graph: Any, requirements, **kwargs) -> Any:
    """
    This type of mutation is passed over all nodes of the tree started from the root node and changes
    nodes’ operations with probability - 'node mutation probability'
    which is initialised inside the function
    """

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probability:
                secondary_node = OptNode(content={'name': choice(requirements.secondary),
                                                  'params': DEFAULT_PARAMS_STUB},
                                         nodes_from=node.nodes_from)
                graph.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probability:
                primary_node = OptNode(content={'name': choice(requirements.primary),
                                                'params': DEFAULT_PARAMS_STUB})
                graph.update_node(node, primary_node)

    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  node=graph.root_node)

    replace_node_to_random_recursive(graph.root_node)

    return graph


def single_edge_mutation(graph: Any, max_depth, *args, **kwargs):
    old_graph = deepcopy(graph)

    for _ in range(MAX_NUM_OF_ATTEMPTS):
        if len(graph.nodes) < 2 or graph.depth > max_depth:
            return graph

        source_node, target_node = sample(graph.nodes, 2)

        nodes_not_cycling = (target_node.descriptive_id not in
                             [n.descriptive_id for n in source_node.ordered_subnodes_hierarchy()])
        if nodes_not_cycling and (target_node.nodes_from is None or source_node not in target_node.nodes_from):
            graph.operator.connect_nodes(source_node, target_node)
            break

    if graph.depth > max_depth:
        return old_graph
    return graph


def _add_intermediate_node(graph: Any, requirements, params, node_to_mutate):
    # add between node and parent
    candidates = params.advisor.propose_parent(str(node_to_mutate.content['name']),
                                               [str(n.content['name']) for n in node_to_mutate.nodes_from],
                                               requirements.secondary)
    if len(candidates) == 0:
        return graph
    new_node = OptNode(content={'name': choice(candidates),
                                'params': DEFAULT_PARAMS_STUB})
    new_node.nodes_from = node_to_mutate.nodes_from
    node_to_mutate.nodes_from = [new_node]
    graph.nodes.append(new_node)
    return graph


def _add_separate_parent_node(graph: Any, requirements, params, node_to_mutate):
    # add as separate parent
    candidates = params.advisor.propose_parent(str(node_to_mutate.content['name']), None,
                                               requirements.primary)
    if len(candidates) == 0:
        return graph
    for iter_num in range(randint(1, 3)):
        if iter_num == len(candidates):
            break
        new_node = OptNode(content={'name': choice(candidates),
                                    'params': DEFAULT_PARAMS_STUB})
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(new_node)
        else:
            node_to_mutate.nodes_from = [new_node]
        graph.nodes.append(new_node)
    return graph


def _add_as_child(graph: Any, requirements, params, node_to_mutate):
    # add as child
    new_node = OptNode(content={'name': choice(requirements.secondary),
                                'params': DEFAULT_PARAMS_STUB})
    new_node.nodes_from = [node_to_mutate]
    graph.operator.actualise_old_node_children(node_to_mutate, new_node)
    graph.nodes.append(new_node)
    return graph


def single_add_mutation(graph: Any, requirements, params, max_depth, *args, **kwargs):
    """
    Add new node between two sequential existing modes
    """

    if graph.depth >= max_depth:
        # add mutation is not possible
        return graph

    node_to_mutate = choice(graph.nodes)

    single_add_strategies = [_add_as_child, _add_separate_parent_node]
    if node_to_mutate.nodes_from:
        single_add_strategies.append(_add_intermediate_node)
    strategy = choice(single_add_strategies)

    result = strategy(graph, requirements, params, node_to_mutate)
    return result


def single_change_mutation(graph: Any, requirements, params, *args, **kwargs):
    """
    Change node between two sequential existing modes
    """
    node = choice(graph.nodes)
    nodes_from = node.nodes_from
    candidates = requirements.secondary if node.nodes_from else requirements.primary
    if params.advisor:
        candidates = params.advisor.propose_change(current_operation_id=str(node.content['name']),
                                                   possible_operations=candidates)

    if len(candidates) == 0:
        return graph

    node_new = OptNode(content={'name': choice(candidates),
                                'params': DEFAULT_PARAMS_STUB})
    node_new.nodes_from = nodes_from
    graph.nodes = [node_new if n == node else n for n in graph.nodes]
    graph.operator.actualise_old_node_children(node, node_new)
    return graph


def single_drop_mutation(graph: Any, params, *args, **kwargs):
    """
    Drop single node from graph
    """
    node_to_del = choice(graph.nodes)
    node_name = node_to_del.content['name']
    removal_type = params.advisor.can_be_removed(str(node_name))
    if removal_type == RemoveType.with_direct_children:
        # TODO refactor workaround with data_source
        nodes_to_delete = \
            [n for n in graph.nodes if str(node_name) in n.descriptive_id and
             n.descriptive_id.count('data_source') == 1]
        for child_node in nodes_to_delete:
            graph.delete_node(child_node)
        graph.delete_node(node_to_del)
    elif removal_type == RemoveType.with_parents:
        graph.delete_subtree(node_to_del)
    elif removal_type != RemoveType.forbidden:
        graph.delete_node(node_to_del)
        if node_to_del.nodes_from:
            childs = graph.operator.node_children(node_to_del)
            for child in childs:
                if child.nodes_from:
                    child.nodes_from.extend(node_to_del.nodes_from)
                else:
                    child.nodes_from = node_to_del.nodes_from
    return graph


def _tree_growth(graph: Any, requirements, params, max_depth: int, local_growth=True):
    """
    This mutation selects a random node in a tree, generates new subtree,
    and replaces the selected node's subtree.
    """
    random_layer_in_graph = randint(0, graph.depth - 1)
    node_from_graph = choice(graph.operator.nodes_from_layer(random_layer_in_graph))
    if local_growth:
        is_primary_node_selected = (not node_from_graph.nodes_from) or (
                node_from_graph.nodes_from and
                node_from_graph != graph.root_node
                and randint(0, 1))
    else:
        is_primary_node_selected = \
            randint(0, 1) and \
            not graph.operator.distance_to_root_level(node_from_graph) < max_depth
    if is_primary_node_selected:
        new_subtree = OptNode(content={'name': choice(requirements.primary),
                                       'params': DEFAULT_PARAMS_STUB})
    else:
        if local_growth:
            max_depth = node_from_graph.distance_to_primary_level
        else:
            max_depth = max_depth - graph.operator.distance_to_root_level(node_from_graph)
        new_subtree = random_graph(params=params, requirements=requirements,
                                   max_depth=max_depth).root_node
    graph.update_subtree(node_from_graph, new_subtree)
    return graph


def growth_mutation(graph: Any, requirements, params, max_depth: int, local_growth=True) -> Any:
    """
    This mutation adds new nodes to the graph (just single node between existing nodes or new subtree).
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """

    if random() > 0.5:
        # simple growth (one node can be added)
        return single_add_mutation(graph, requirements, params, max_depth)
    else:
        # advanced growth (several nodes can be added)
        return _tree_growth(graph, requirements, params, max_depth, local_growth)


def reduce_mutation(graph: OptGraph, requirements, **kwargs) -> OptGraph:
    """
    Selects a random node in a tree, then removes its subtree. If the current arity of the node's
    parent is more than the specified minimal arity, then the selected node is also removed.
    Otherwise, it is replaced by a random primary node.
    """
    if len(graph.nodes) == 1:
        return graph

    nodes = [node for node in graph.nodes if node is not graph.root_node]
    node_to_del = choice(nodes)
    children = graph.operator.node_children(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in children])
    if is_possible_to_delete:
        graph.delete_subtree(node_to_del)
    else:
        primary_node = OptNode(content={'name': choice(requirements.primary),
                                        'params': DEFAULT_PARAMS_STUB})
        graph.update_subtree(node_to_del, primary_node)
    return graph


mutation_by_type = {
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.reduce: reduce_mutation,
    MutationTypesEnum.single_add: single_add_mutation,
    MutationTypesEnum.single_edge: single_edge_mutation,
    MutationTypesEnum.single_drop: single_drop_mutation,
    MutationTypesEnum.single_change: single_change_mutation,

}
