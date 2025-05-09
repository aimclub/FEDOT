from random import choice, random
from typing import List

from golem.core.optimisers.genetic.operators.base_mutations import get_mutation_prob

from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.hyperparams import ParametersChanger
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


def parameter_change_mutation(pipeline: Pipeline, requirements, graph_gen_params, parameters, **kwargs) -> Pipeline:
    """
    This type of mutation is passed over all nodes and changes
    hyperparameters of the operations with probability - 'node mutation probability'
    which is initialised inside the function
    """
    node_mutation_probability = get_mutation_prob(mut_id=parameters.mutation_strength,
                                                  node=pipeline.root_node)
    for node in pipeline.nodes:
        lagged = node.operation.metadata.id in ('lagged', 'sparse_lagged', 'exog_ts')
        do_mutation = random() < (node_mutation_probability * (0.5 if lagged else 1))
        if do_mutation:
            operation_name = node.operation.operation_type
            current_params = node.parameters

            # Perform specific change for particular parameter
            changer = ParametersChanger(operation_name, current_params)
            try:
                new_params = changer.get_new_operation_params()
                if new_params is not None:
                    node.parameters = new_params
            except Exception as ex:
                pipeline.log.error(ex)
    return pipeline


def boosting_mutation(pipeline: Pipeline, requirements, graph_gen_params, **kwargs) -> Pipeline:
    """ This type of mutation adds the additional 'boosting' cascade to the existing pipeline """

    # TODO: refactor next line to get task_type more obviously
    task_type = graph_gen_params.advisor.task.task_type
    decompose_operations = OperationTypesRepository('data_operation').suitable_operation(
        task_type=task_type, tags=['decompose'])
    decompose_operation = decompose_operations[0]

    existing_pipeline = pipeline

    all_data_operations = OperationTypesRepository('data_operation').suitable_operation(
        task_type=task_type)
    preprocessing_primary_nodes = [n for n in existing_pipeline.nodes if str(n) in all_data_operations]

    if len(preprocessing_primary_nodes) > 0:
        data_source = preprocessing_primary_nodes[0]
    else:
        if task_type == TaskTypesEnum.ts_forecasting:
            data_source = PipelineNode('simple_imputation')
        else:
            data_source = PipelineNode('scaling')

    decompose_parents = [existing_pipeline.root_node, data_source]

    boosting_model_candidates = requirements.secondary
    if task_type == TaskTypesEnum.classification:
        # the regression models are required
        boosting_model_candidates = \
            OperationTypesRepository('model').suitable_operation(
                task_type=TaskTypesEnum.regression, forbidden_tags=['non_lagged'])
        if not boosting_model_candidates:
            return pipeline

    new_model = choose_new_model(boosting_model_candidates)

    if task_type == TaskTypesEnum.ts_forecasting:
        non_lagged_ts_models = OperationTypesRepository('model').operations_with_tag(['non_lagged'])
        is_non_lagged_ts_models_in_node = \
            str(existing_pipeline.root_node) in non_lagged_ts_models

        if is_non_lagged_ts_models_in_node:
            # if additional lagged node is required
            lagged_node = PipelineNode('lagged', nodes_from=[data_source])
            decompose_parents = [existing_pipeline.root_node, lagged_node]

    node_decompose = PipelineNode(decompose_operation, nodes_from=decompose_parents)

    node_boost = PipelineNode(new_model, nodes_from=[node_decompose])

    node_final = PipelineNode(choice(requirements.secondary),
                              nodes_from=[existing_pipeline.root_node, node_boost])
    pipeline = Pipeline(node_final, use_input_preprocessing=pipeline.use_input_preprocessing)
    return pipeline


def add_resample_mutation(pipeline: Pipeline, **kwargs):
    """
    Add resample operation before all primary operations in pipeline

    :param pipeline: pipeline to insert resample

    :return: mutated pipeline
    """
    resample_node = PipelineNode('resample')

    p_nodes = [p_node for p_node in pipeline.primary_nodes]
    pipeline.add_node(resample_node)

    for node in p_nodes:
        pipeline.connect_nodes(resample_node, node)
    return pipeline


def choose_new_model(boosting_model_candidates: List[str]) -> str:
    """ Since 'ridge' and 'rfr' operations are suitable for solving the problem
    and they are simpler than others, they are preferred """

    if 'ridge' in boosting_model_candidates:
        new_model = 'ridge'
    elif 'rfr' in boosting_model_candidates:
        new_model = 'rfr'
    else:
        new_model = choice(boosting_model_candidates)
    return new_model
