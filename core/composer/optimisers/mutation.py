from copy import deepcopy
from enum import Enum
from random import random, choice
from typing import Any

from core.composer.chain import Chain
from core.composer.optimisers.gp_operators import node_depth


class MutationPowerEnum(Enum):
    weak = 0
    mean = 1
    strong = 2


def get_mutation_prob(mut_id, root_node):
    default_mutation_prob = 0.7
    if mut_id == MutationPowerEnum.weak.value:
        return 1.0 / (5.0 * (node_depth(root_node) + 1))
    elif mut_id == MutationPowerEnum.mean.value:
        return 1.0 / (node_depth(root_node) + 1)
    elif mut_id == MutationPowerEnum.strong.value:
        return 5.0 / (node_depth(root_node) + 1)
    else:
        return default_mutation_prob


def standard_mutation(chain: Chain, secondary: Any, primary: Any,
                      secondary_node_func: Any = None, primary_node_func: Any = None, mutation_prob: bool = 0.8,
                      node_mutate_type=MutationPowerEnum.mean) -> Any:
    result = deepcopy(chain)
    if mutation_prob and random() > mutation_prob:
        return result

    node_mutation_probality = get_mutation_prob(mut_id=node_mutate_type.value, root_node=result.root_node)

    def replace_node_to_random_recursive(node: Any) -> Any:
        if node.nodes_from:
            if random() < node_mutation_probality:
                secondary_node = secondary_node_func(model_type=choice(secondary), nodes_from=node.nodes_from)
                result.update_node(node, secondary_node)
            for child in node.nodes_from:
                replace_node_to_random_recursive(child)
        else:
            if random() < node_mutation_probality:
                primary_node = primary_node_func(model_type=choice(primary))
                result.update_node(node, primary_node)

    replace_node_to_random_recursive(result.root_node)

    return result
