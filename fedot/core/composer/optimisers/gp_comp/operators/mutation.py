from copy import deepcopy
from functools import partial
from random import choice, randint, random
from typing import Any

from fedot.core.chains.chain import Chain, List
from fedot.core.chains.tuning.hyperparams import get_new_operation_params
from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_comp.gp_operators import node_depth, node_height, nodes_from_height, random_chain
from fedot.core.log import Log
from fedot.core.utils import ComparableEnum as Enum

MAX_NUM_OF_ATTEMPTS = 10


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    parameter_change = 'parameter_change'
    none = 'none'


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


def get_mutation_prob(mut_id, root_node):
    """ Function returns mutation probability for certain node in the chain

    :param mut_id: MutationStrengthEnum mean weak or strong mutation
    :param root_node: root node of the chain
    :return mutation_prob: mutation probability
    """

    default_mutation_prob = 0.7
    if mut_id in list(MutationStrengthEnum):
        mutation_strength = mut_id.value
        mutation_prob = mutation_strength / (node_depth(root_node) + 1)
    else:
        mutation_prob = default_mutation_prob
    return mutation_prob


def will_mutation_be_applied(mutation_prob, mutation_type) -> bool:
    return not (random() > mutation_prob or mutation_type == MutationTypesEnum.none)


def mutation(types: List[MutationTypesEnum], chain_generation_params, chain: Chain, requirements, log: Log,
             max_depth: int = None) -> Any:
    """ Function apply mutation operator to chain """
    max_depth = max_depth if max_depth else requirements.max_depth
    mutation_prob = requirements.mutation_prob
    mutation_type = choice(types)
    if will_mutation_be_applied(mutation_prob, mutation_type):
        if mutation_type in mutation_by_type:
            for _ in range(MAX_NUM_OF_ATTEMPTS):
                new_chain = mutation_by_type[mutation_type](chain=deepcopy(chain), requirements=requirements,
                                                            chain_generation_params=chain_generation_params,
                                                            max_depth=max_depth)
                is_correct_chain = constraint_function(new_chain)
                if is_correct_chain:
                    return new_chain
        elif mutation_type != MutationTypesEnum.none:
            raise ValueError(f'Required mutation type is not found: {mutation_type}')
        log.debug('Number of mutation attempts exceeded. Please check composer requirements for correctness.')
    return deepcopy(chain)


def simple_mutation(chain: Any, requirements, chain_generation_params, max_depth: int = None) -> Any:
    """
    This type of mutation is passed over all nodes of the tree started from the root node and changes
    nodes’ operations with probability - 'node mutation probability' which is inicialised inside the function
    """

    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  root_node=chain.root_node)

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probability:
                secondary_node = chain_generation_params.secondary_node_func(choice(requirements.secondary),
                                                                             nodes_from=node.nodes_from)
                chain.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probability:
                primary_node = chain_generation_params.primary_node_func(operation_type=choice(requirements.primary))
                chain.update_node(node, primary_node)

    replace_node_to_random_recursive(chain.root_node)

    return chain


def growth_mutation(chain: Any, requirements, chain_generation_params, max_depth: int, local_growth=True) -> Any:
    """
    This mutation selects a random node in a tree, generates new subtree, and replaces the selected node's subtree.
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to new subtree depth,
    maximal depth of new subtree just should satisfy depth constraint in parent tree
    """

    random_layer_in_chain = randint(0, chain.depth - 1)
    node_from_chain = choice(nodes_from_height(chain, random_layer_in_chain))
    if local_growth:
        is_primary_node_selected = (not node_from_chain.nodes_from) or (
                node_from_chain.nodes_from and node_from_chain != chain.root_node and randint(0, 1))
    else:
        is_primary_node_selected = randint(0, 1) and not node_height(chain, node_from_chain) < max_depth
    if is_primary_node_selected:
        new_subtree = chain_generation_params.primary_node_func(operation_type=choice(requirements.primary))
    else:
        if local_growth:
            max_depth = node_depth(node_from_chain)
        else:
            max_depth = max_depth - node_height(chain, node_from_chain)
        new_subtree = random_chain(chain_generation_params=chain_generation_params, requirements=requirements,
                                   max_depth=max_depth).root_node
    chain.replace_node_with_parents(node_from_chain, new_subtree)
    return chain


def reduce_mutation(chain: Any, requirements, chain_generation_params, max_depth: int = None) -> Any:
    """
    Selects a random node in a tree, then removes its subtree. If the current arity of the node's
    parent is more than the specified minimal arity, then the selected node is also removed.
    Otherwise, it is replaced by a random primary node.
    """

    nodes = [node for node in chain.nodes if node is not chain.root_node]
    node_to_del = choice(nodes)
    childs = chain.node_childs(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in childs])
    if is_possible_to_delete:
        chain.delete_subtree(node_to_del)
    else:
        primary_node = chain_generation_params.primary_node_func(operation_type=choice(requirements.primary))
        chain.replace_node_with_parents(node_to_del, primary_node)
    return chain


def parameter_change_mutation(chain: Any, requirements, **kwargs) -> Any:
    """
    This type of mutation is passed over all nodes and changes
    hyperpearameters of the operations with probability - 'node mutation probability'
    which is initialised inside the function
    """
    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  root_node=chain.root_node)
    for node in chain.nodes:
        if random() < node_mutation_probability:
            operation_name = node.operation.operation_type
            node.custom_params = get_new_operation_params(operation_name)

    return chain


mutation_by_type = {
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.reduce: reduce_mutation,
    MutationTypesEnum.parameter_change: parameter_change_mutation
}
