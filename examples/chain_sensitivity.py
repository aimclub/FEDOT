from os import makedirs
from os.path import join, exists

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root, default_fedot_data_dir
from fedot.sensitivity.chain_sensitivity import ChainStructureAnalyze
from fedot.sensitivity.node_sensitivity import NodeDeletionAnalyze, NodeReplaceOperationAnalyze


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


def get_scoring_data():
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = join(str(project_root()), file_path_test)
    task = Task(TaskTypesEnum.classification)
    train = InputData.from_csv(full_path_train, task=task)
    test = InputData.from_csv(full_path_test, task=task)

    return train, test


def get_kc2_data():
    file_path = 'cases/data/kc2/kc2.csv'
    full_path = join(str(project_root()), file_path)
    task = Task(TaskTypesEnum.classification)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


def get_cholesterol_data():
    file_path = 'cases/data/cholesterol/cholesterol.csv'
    full_path = join(str(project_root()), file_path)
    task = Task(TaskTypesEnum.regression)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


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


def run_analysis_case(train_data: InputData, test_data: InputData,
                      case_name: str, task, metric, is_composed=False, result_path=None):
    chain = chain_by_task(task=task, metric=metric,
                          data=train_data, is_composed=is_composed)

    chain.fit(train_data)

    if not result_path:
        result_path = join(default_fedot_data_dir(), 'sensitivity', f'{case_name}')
        if not exists(result_path):
            makedirs(result_path)

    visualiser = ChainVisualiser()
    visualiser.visualise(chain, save_path=result_path)

    chain_analysis_result = ChainStructureAnalyze(chain=chain, train_data=train_data,
                                                  test_data=test_data, all_nodes=True, path_to_save=result_path,
                                                  approaches=[NodeDeletionAnalyze,
                                                              NodeReplaceOperationAnalyze]).analyze()

    print(f'chain analysis result {chain_analysis_result}')


def run_class_scoring_case(is_composed: bool, path_to_save=None):
    train_data, test_data = get_scoring_data()
    task = Task(TaskTypesEnum.classification)
    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    if is_composed:
        case = 'scoring_composed'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=True, result_path=path_to_save)
    else:
        case = 'scoring'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=False, result_path=path_to_save)


def run_class_kc2_case(is_composed: bool = False, path_to_save=None):
    train_data, test_data = get_kc2_data()
    task = Task(TaskTypesEnum.classification)
    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    if is_composed:
        case = 'kc2_composed'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=True, result_path=path_to_save)
    else:
        case = 'kc2'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=False, result_path=path_to_save)


def run_regr_case(is_composed: bool = False, path_to_save=None):
    train_data, test_data = get_cholesterol_data()
    task = Task(TaskTypesEnum.regression)
    # the choice of the metric for the chain quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    if is_composed:
        case = 'cholesterol_composed'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=True, result_path=path_to_save)
    else:
        case = 'cholesterol'
        run_analysis_case(train_data, test_data, case, task,
                          metric=metric_function,
                          is_composed=False, result_path=path_to_save)


if __name__ == '__main__':
    # scoring case manual
    run_class_scoring_case(is_composed=False)

    # kc2 case manual
    run_class_kc2_case(is_composed=False)

    # cholesterol regr case
    run_regr_case(is_composed=False)
