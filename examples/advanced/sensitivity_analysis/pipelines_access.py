from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task


def get_three_depth_manual_class_pipeline():
    logit_node_primary = PrimaryNode('logit')
    xgb_node_primary = PrimaryNode('xgboost')
    xgb_node_primary_second = PrimaryNode('xgboost')

    qda_node_third = SecondaryNode('qda', nodes_from=[xgb_node_primary_second])
    knn_node_third = SecondaryNode('knn', nodes_from=[logit_node_primary, xgb_node_primary])

    knn_root = SecondaryNode('knn', nodes_from=[qda_node_third, knn_node_third])

    pipeline = Pipeline(knn_root)

    return pipeline


def get_three_depth_manual_regr_pipeline():
    rfr_primary = PrimaryNode('rfr')
    knn_primary = PrimaryNode('knnreg')

    dtreg_secondary = SecondaryNode('dtreg', nodes_from=[rfr_primary])
    rfr_secondary = SecondaryNode('rfr', nodes_from=[knn_primary])

    knnreg_root = SecondaryNode('knnreg', nodes_from=[dtreg_secondary, rfr_secondary])

    pipeline = Pipeline(knnreg_root)

    return pipeline


def get_composed_pipeline(dataset_to_compose, task, metric_function):
    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='model')

    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
    )
    params = GPGraphOptimizerParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state
    )

    # Create composer and with required composer params
    composer = ComposerBuilder(task=task). \
        with_requirements(composer_requirements). \
        with_optimizer_params(params). \
        with_metrics(metric_function). \
        build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)

    return pipeline_evo_composed


def pipeline_by_task(task, metric, data, is_composed):
    if is_composed:
        pipeline = get_composed_pipeline(data, task,
                                         metric_function=metric)
    else:
        if task.task_type.name == 'classification':
            pipeline = get_three_depth_manual_class_pipeline()
        else:
            pipeline = get_three_depth_manual_regr_pipeline()

    return pipeline
