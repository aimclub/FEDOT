from copy import deepcopy
from functools import partial
from random import choice, randint, random
from typing import (Any, Callable)

from fedot.core.composer.chain import Chain, List
from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_operators import node_depth, nodes_from_height, random_chain
from fedot.core.utils import ComparableEnum as Enum


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    reduce = 'reduce'


class MutationPowerEnum(Enum):
    weak = 0
    mean = 1
    strong = 2


def get_mutation_prob(mut_id, root_node):
    default_mutation_prob = 0.7
    if mut_id == MutationPowerEnum.weak.value:
        mutation_strength = 0.2
        return mutation_strength / (node_depth(root_node) + 1)
    elif mut_id == MutationPowerEnum.mean.value:
        mutation_strength = 1.0
        return mutation_strength / (node_depth(root_node) + 1)
    elif mut_id == MutationPowerEnum.strong.value:
        mutation_strength = 5.0
        return mutation_strength / (node_depth(root_node) + 1)
    else:
        return default_mutation_prob


def mutation(types: List[MutationTypesEnum], chain_class, chain: Chain, requirements,
             secondary_node_func: Callable = None, primary_node_func: Callable = None, mutation_prob: bool = 0.8,
             node_mutate_type=MutationPowerEnum.mean) -> Any:
    if mutation_prob and random() > mutation_prob:
        return deepcopy(chain)

    mutation_by_type = {
        MutationTypesEnum.simple: partial(simple_mutation, secondary=requirements.secondary,
                                          primary=requirements.primary,
                                          secondary_node_func=secondary_node_func,
                                          primary_node_func=primary_node_func,
                                          node_mutate_type=node_mutate_type),
        MutationTypesEnum.growth: partial(growth_mutation, chain_class=chain_class,
                                          secondary_node_func=secondary_node_func,
                                          primary_node_func=primary_node_func, requirements=requirements),
        MutationTypesEnum.reduce: partial(reduce_mutation, primary_node_func=primary_node_func,
                                          requirements=requirements)
    }
    type = choice(types)
    if type in mutation_by_type:
        is_correct_chain = False
        while not is_correct_chain:
            new_chain = mutation_by_type[type](chain=chain)
            is_correct_chain = constraint_function(new_chain)
        return new_chain
    else:
        raise ValueError(f'Required mutation not found: {type}')


def simple_mutation(chain: Any, secondary: Any, primary: Any,
                    secondary_node_func: Callable, primary_node_func: Callable,
                    node_mutate_type=MutationPowerEnum.mean) -> Any:
    result = deepcopy(chain)

    node_mutation_probability = get_mutation_prob(mut_id=node_mutate_type.value,
                                                  root_node=result.root_node)

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probability:
                secondary_node = secondary_node_func(model_type=choice(secondary),
                                                     nodes_from=node.nodes_from)
                result.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probability:
                primary_node = primary_node_func(model_type=choice(primary))
                result.update_node(node, primary_node)

    replace_node_to_random_recursive(result.root_node)

    return result


def growth_mutation(chain: Any, chain_class, secondary_node_func: Callable, primary_node_func: Callable,
                    requirements) -> Any:
    chain_copy = deepcopy(chain)
    random_layer_in_chain = randint(0, chain_copy.depth - 1)
    node_from_chain = choice(nodes_from_height(chain_copy, random_layer_in_chain))
    is_primary_node_selected = (not node_from_chain.nodes_from) or (
            node_from_chain.nodes_from and node_from_chain != chain_copy.root_node and randint(0, 1))
    if is_primary_node_selected:
        new_subtree = primary_node_func(model_type=choice(requirements.primary))
    else:
        max_depth = node_depth(node_from_chain)
        new_subtree = random_chain(chain_class, secondary_node_func, primary_node_func, requirements,
                                   max_depth=max_depth).root_node
    chain_copy.replace_node_with_parents(node_from_chain, new_subtree)
    return chain_copy


def reduce_mutation(chain: Any, primary_node_func: Callable, requirements) -> Any:
    chain_copy = deepcopy(chain)
    nodes = [node for node in chain_copy.nodes if not node is chain_copy.root_node]
    node_to_del = choice(nodes)
    childs = chain_copy.node_childs(node_to_del)
    is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in childs])
    if is_possible_to_delete:
        chain_copy.delete_node(node_to_del)
    else:
        primary_node = primary_node_func(model_type=choice(requirements.primary))
        chain_copy.replace_node_with_parents(node_to_del, primary_node)
    return chain_copy
