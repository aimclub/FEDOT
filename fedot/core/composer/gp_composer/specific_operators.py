from random import choice, random
from typing import Any

from fedot.core.optimisers.gp_comp.operators.mutation import get_mutation_prob
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.hyperparams import ParametersChanger
from fedot.core.repository.operation_types_repository import OperationTypesRepository


def parameter_change_mutation(pipeline: Pipeline, requirements, **kwargs) -> Any:
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
            current_params = node.operation.params

            # Perform specific change for particular parameter
            changer = ParametersChanger(operation_name, current_params)
            node.custom_params = changer.get_new_operation_params()

    return pipeline


def boosting_mutation(pipeline: Pipeline, requirements, params, **kwargs) -> Any:
    """
    This type of mutation adds the additional 'boosting' cascade to the existing pipeline.
    """

    task_type = params.advisor.task.task_type
    decompose_operations, _ = OperationTypesRepository('data_operation').suitable_operation(
        task_type=task_type, tags=['decompose'])
    decompose_operation = decompose_operations[0]

    existing_pipeline = pipeline

    if len(pipeline.nodes) == 1:
        # to deal with single-node pipeline
        data_source = pipeline.nodes[0]
    else:
        data_source = PrimaryNode('scaling')

    decompose_parents = [existing_pipeline.root_node, data_source]

    node_decompose = SecondaryNode(decompose_operation, nodes_from=decompose_parents)
    node_boost = SecondaryNode('linear', nodes_from=[node_decompose])
    node_final = SecondaryNode(choice(requirements.secondary),
                               nodes_from=[node_boost, existing_pipeline.root_node])
    pipeline.nodes.extend([node_decompose, node_final, node_boost])
    return pipeline
