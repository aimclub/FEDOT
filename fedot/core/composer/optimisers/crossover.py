from copy import deepcopy
from random import choice, random
from typing import Any, List

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_operators import \
    (equivalent_subtree, node_depth,
     nodes_from_height, replace_subtrees)
from fedot.core.utils import ComparableEnum as Enum


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


def crossover(types: List[CrossoverTypesEnum], chain_first: Any, chain_second: Any, max_depth: int,
              crossover_prob: float = 0.8) -> Any:
    type = choice(types)
    chain_first_copy = deepcopy(chain_first)
    chain_second_copy = deepcopy(chain_second)
    try:
        if chain_first is chain_second or random() > crossover_prob or type == CrossoverTypesEnum.none:
            return [chain_first_copy, chain_second_copy]
        if type in crossover_by_type.keys():
            is_correct = False
            while not is_correct:
                is_correct_chains = []
                new_chains = crossover_by_type[type](chain_first_copy, chain_second_copy, max_depth)
                for new_chain in new_chains:
                    is_correct_chains.append(constraint_function(new_chain))
                is_correct = all(is_correct_chains)
            return new_chains
        else:
            raise ValueError(f'Required crossover not found: {type}')
    except Exception as ex:
        print(f'Crossover ex: {ex}')
        return chain_first_copy, chain_second_copy


def subtree_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    """Performed by the replacement of random subtree
    in first selected parent to random subtree from the second parent"""
    random_layer_in_chain_first = choice(range(chain_first.depth))
    min_second_layer = 1 if random_layer_in_chain_first == 0 else 0
    random_layer_in_chain_second = choice(range(min_second_layer, chain_second.depth))

    node_from_chain_first = choice(nodes_from_height(chain_first, random_layer_in_chain_first))
    node_from_chain_second = choice(nodes_from_height(chain_second, random_layer_in_chain_second))

    replace_subtrees(chain_first, chain_second, node_from_chain_first, node_from_chain_second,
                     random_layer_in_chain_first, random_layer_in_chain_second, max_depth)

    return chain_first, chain_second


def one_point_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    """Finds common structural parts between two trees, and after that randomly
    chooses the location of nodes, subtrees of which will be swapped"""
    pairs_of_nodes = equivalent_subtree(chain_first, chain_second)
    if pairs_of_nodes:
        node_from_chain_first, node_from_chain_second = choice(pairs_of_nodes)

        layer_in_chain_first = node_depth(chain_first.root_node) - node_depth(node_from_chain_first)
        layer_in_chain_second = node_depth(chain_second.root_node) - node_depth(node_from_chain_second)

        replace_subtrees(chain_first, chain_second, node_from_chain_first, node_from_chain_second,
                         layer_in_chain_first, layer_in_chain_second, max_depth)
    return chain_first, chain_second


crossover_by_type = {
    CrossoverTypesEnum.subtree: subtree_crossover,
    CrossoverTypesEnum.one_point: one_point_crossover,
}
