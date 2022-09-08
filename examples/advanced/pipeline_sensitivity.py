from os import makedirs
from os.path import exists, join

from examples.simple.classification.classification_pipelines import classification_three_depth_manual_pipeline
from examples.simple.regression.regression_pipelines import regression_three_depth_manual_pipeline
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import default_fedot_data_dir, fedot_project_root
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis


def get_composed_pipeline(dataset_to_compose, task, metric_function):
    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='model')

    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
        max_arity=3, max_depth=3,
        num_of_generations=20,
    )

    optimizer_parameters = GPGraphOptimizerParameters(
        pop_size=15,
        mutation_prob=0.8, crossover_prob=0.8,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
    )

    # Create composer and with required composer params
    composer = ComposerBuilder(task=task). \
        with_requirements(composer_requirements). \
        with_optimizer_params(optimizer_parameters). \
        with_metrics(metric_function). \
        build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)

    return pipeline_evo_composed


def get_scoring_data():
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = join(str(fedot_project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = join(str(fedot_project_root()), file_path_test)
    task = Task(TaskTypesEnum.classification)
    train = InputData.from_csv(full_path_train, task=task)
    test = InputData.from_csv(full_path_test, task=task)

    return train, test


def get_kc2_data():
    file_path = 'cases/data/kc2/kc2.csv'
    full_path = join(str(fedot_project_root()), file_path)
    task = Task(TaskTypesEnum.classification)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


def get_cholesterol_data():
    file_path = 'cases/data/cholesterol/cholesterol.csv'
    full_path = join(str(fedot_project_root()), file_path)
    task = Task(TaskTypesEnum.regression)
    data = InputData.from_csv(full_path, task=task)
    train, test = train_test_data_setup(data)

    return train, test


def pipeline_by_task(task, metric, data, is_composed):
    if is_composed:
        pipeline = get_composed_pipeline(data, task,
                                         metric_function=metric)
    else:
        if task.task_type.name == 'classification':
            pipeline = classification_three_depth_manual_pipeline()
        else:
            pipeline = regression_three_depth_manual_pipeline()

    return pipeline


def run_analysis_case(train_data: InputData, test_data: InputData,
                      case_name: str, task, metric, is_composed=False, result_path=None):
    pipeline = pipeline_by_task(task=task, metric=metric,
                                data=train_data, is_composed=is_composed)

    pipeline.fit(train_data)

    if not result_path:
        result_path = join(default_fedot_data_dir(), 'sensitivity', f'{case_name}')
        if not exists(result_path):
            makedirs(result_path)

    pipeline.show(save_path=result_path)

    pipeline_analysis_result = NodesAnalysis(pipeline=pipeline, train_data=train_data,
                                             test_data=test_data, path_to_save=result_path,
                                             approaches=[NodeDeletionAnalyze,
                                                         NodeReplaceOperationAnalyze]).analyze()

    print(f'pipeline analysis result {pipeline_analysis_result}')


def run_class_scoring_case(is_composed: bool, path_to_save=None):
    train_data, test_data = get_scoring_data()
    task = Task(TaskTypesEnum.classification)
    # the choice of the metric for the pipeline quality assessment during composition
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
    # the choice of the metric for the pipeline quality assessment during composition
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
    # the choice of the metric for the pipeline quality assessment during composition
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
