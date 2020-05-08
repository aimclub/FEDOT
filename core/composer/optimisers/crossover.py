from copy import deepcopy
from enum import Enum
from random import randint, choice
from random import random
from typing import Any, List

from core.composer.optimisers.gp_operators import nodes_from_height, node_depth


class CrossoverTypesEnum(Enum):
    subtree = 'subtree'


def crossover(types: List[CrossoverTypesEnum], chain_first: Any, chain_second: Any, max_depth: int,
              crossover_prob: float = 0.8) -> Any:
    if chain_first is chain_second or random() > crossover_prob:
        return deepcopy(chain_first)
    type = choice(types)
    if type in crossover_by_type.keys():
        return crossover_by_type[type](chain_first, chain_second, max_depth)
    else:
        raise ValueError(f'Required crossover not found: {type}')


def subtree_crossover(chain_first: Any, chain_second: Any, max_depth: int) -> Any:
    chain_first_copy = deepcopy(chain_first)
    random_layer_in_chain_first = randint(0, chain_first_copy.depth - 1)
    random_layer_in_chain_second = randint(0, chain_second.depth - 1)
    if random_layer_in_chain_first == 0 and random_layer_in_chain_second == 0:
        if randint(0, 1):
            random_layer_in_chain_first = randint(1, chain_first_copy.depth - 1)
        else:
            random_layer_in_chain_second = randint(1, chain_second.depth - 1)

    node_from_chain_first = choice(nodes_from_height(chain_first_copy, random_layer_in_chain_first))
    node_from_chain_second = choice(nodes_from_height(chain_second, random_layer_in_chain_second))

    summary_depth = random_layer_in_chain_first + node_depth(node_from_chain_second)
    if summary_depth <= max_depth and summary_depth != 0:
        chain_first_copy.replace_node_with_parents(node_from_chain_first, node_from_chain_second)

    return chain_first_copy


crossover_by_type = {
    CrossoverTypesEnum.subtree: subtree_crossover}
