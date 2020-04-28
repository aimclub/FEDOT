import random
from copy import deepcopy
from random import randint, choice
from typing import Any
from enum import Enum

from core.composer.optimisers.gp_operators import nodes_from_height, node_depth


class CrossoverTypeEnum(Enum):
    standard = 0


def crossover(cros_type, chain_first: Any, chain_second: Any, max_depth: int, crossover_prob: float = 0.8) -> Any:
    if cros_type == CrossoverTypeEnum.standard:
        return standard_crossover(chain_first, chain_second, max_depth, crossover_prob)


def standard_crossover(chain_first: Any, chain_second: Any, max_depth: int, crossover_prob: float) -> Any:
    if chain_first is chain_second or random.random() > crossover_prob:
        return deepcopy(chain_first)
    chain_first_copy = deepcopy(chain_first)
    random_layer_in_chain_first = randint(0, chain_first_copy.depth - 1)
    random_layer_in_chain_second = randint(0, chain_second.depth - 1)
    if random_layer_in_chain_first == 0 and random_layer_in_chain_second == 0:
        return deepcopy(chain_second)

    node_from_chain_first = choice(nodes_from_height(chain_first_copy, random_layer_in_chain_first))
    node_from_chain_second = choice(nodes_from_height(chain_second, random_layer_in_chain_second))

    summary_depth = random_layer_in_chain_first + node_depth(node_from_chain_second)
    if summary_depth <= max_depth and summary_depth != 0:
        chain_first_copy.replace_node_with_parents(node_from_chain_first, node_from_chain_second)
        return chain_first_copy
    else:
        return chain_first_copy
