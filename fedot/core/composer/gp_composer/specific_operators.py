from random import random
from typing import Any

from fedot.core.optimisers.gp_comp.operators.mutation import get_mutation_prob
from fedot.core.pipelines.tuning.hyperparams import get_new_operation_params


def parameter_change_mutation(pipeline: Any, requirements, **kwargs) -> Any:
    """
    This type of mutation is passed over all nodes and changes
    hyperparameters of the operations with probability - 'node mutation probability'
    which is initialised inside the function
    """
    node_mutation_probability = get_mutation_prob(mut_id=requirements.mutation_strength,
                                                  node=pipeline.root_node)
    for node in pipeline.nodes:
        if random() < node_mutation_probability:
            operation_name = node.operation.operation_type
            node.custom_params = get_new_operation_params(operation_name)

    return pipeline
