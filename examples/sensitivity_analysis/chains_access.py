from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task


def get_three_depth_manual_class_chain():
    logit_node_primary = PrimaryNode('logit')
    xgb_node_primary = PrimaryNode('xgboost')
    xgb_node_primary_second = PrimaryNode('xgboost')

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_primary_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_primary, xgb_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    chain = Chain(knn_root)

    return chain


def get_three_depth_manual_regr_chain():
    xgb_primary = PrimaryNode('xgbreg')
    knn_primary = PrimaryNode('knnreg')

    dtreg_secondary = SecondaryNode('dtreg', nodes_from=[xgb_primary])
    rfr_secondary = SecondaryNode('rfr', nodes_from=[knn_primary])

    knnreg_root = SecondaryNode('knnreg', nodes_from=[dtreg_secondary, rfr_secondary])

    chain = Chain(knnreg_root)

    return chain


def get_composed_chain(dataset_to_compose, task, metric_function):
    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types = get_operations_for_task(task=task, mode='models')

    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.8, allow_single_operations=False)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_optimiser_parameters(optimiser_parameters)

    # Create GP-based composer
    composer = builder.build()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                is_visualise=True)

    return chain_evo_composed


def chain_by_task(task, metric, data, is_composed):
    if is_composed:
        chain = get_composed_chain(data, task,
                                   metric_function=metric)
    else:
        if task.task_type.name == 'classification':
            chain = get_three_depth_manual_class_chain()
        else:
            chain = get_three_depth_manual_regr_chain()

    return chain
