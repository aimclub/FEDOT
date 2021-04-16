from copy import deepcopy
from random import choice, random
from typing import Any, List

from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.optimisers.gp_comp.gp_operators import \
    (equivalent_subtree, node_depth,
     nodes_from_height, replace_subtrees)
from fedot.core.log import Log
from fedot.core.utils import ComparableEnum as Enum

MAX_NUM_OF_ATTEMPTS = 10


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'
    one_point = "one_point"
    none = 'none'


def will_crossover_be_applied(chain_first, chain_second, crossover_prob, crossover_type) -> bool:
    return not (chain_first is chain_second or
                random() > crossover_prob or
                crossover_type == CrossoverTypesEnum.none)


def crossover(types: List[CrossoverTypesEnum], chain_first: Any, chain_second: Any, max_depth: int, log: Log,
              crossover_prob: float = 0.8) -> Any:
    crossover_type = choice(types)
    try:
        if will_crossover_be_applied(chain_first, chain_second, crossover_prob, crossover_type):
            if crossover_type in crossover_by_type.keys():
                for _ in range(MAX_NUM_OF_ATTEMPTS):
                    new_chains = crossover_by_type[crossover_type](deepcopy(chain_first),
                                                                   deepcopy(chain_second), max_depth)
                    are_correct = all([constraint_function(new_chain) for new_chain in new_chains])
                    if are_correct:
                        return new_chains
            else:
                raise ValueError(f'Required crossover type not found: {crossover_type}')

            log.debug('Number of crossover attempts exceeded. '
                      'Please check composer requirements for correctness.')
    except Exception as ex:
        log.error(f'Crossover ex: {ex}')

    chain_first_copy = deepcopy(chain_first)
    chain_second_copy = deepcopy(chain_second)
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
