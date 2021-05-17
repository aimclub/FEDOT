from random import random
from typing import Any

from fedot.core.chains.tuning.hyperparams import get_new_operation_params
from fedot.core.optimisers.gp_comp.operators.mutation import get_mutation_prob


def parameter_change_mutation(chain: Any, requirements, **kwargs) -> Any:
    """
    This type of mutation is passed over all nodes and changes
    hyperparameters of the operations with probability - 'node mutation probability'
    which is initialised inside the function
    """
    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  node=chain.root_node)
    for node in chain.nodes:
        if random() < node_mutation_probability:
            operation_name = node.operation.operation_type
            node.custom_params = get_new_operation_params(operation_name)

    return chain
