from examples.advanced.sensitivity_analysis.dataset_access import get_scoring_data, get_kc2_data, get_cholesterol_data
from fedot.core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from examples.advanced.sensitivity_analysis.case_analysis import run_case_analysis


def run_class_scoring_case(sa_class: str, is_composed: bool = False, path_to_save=None):
    train_data, test_data = get_scoring_data()
    task = Task(TaskTypesEnum.classification)
    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    if is_composed:
        case = f'scoring_composed_{sa_class}'
        is_composed = True
    else:
        case = f'scoring_{sa_class}'
        is_composed = False

    run_case_analysis(train_data=train_data,
                      test_data=test_data,
                      case_name=case, task=task, sa_class=sa_class,
                      metric=metric_function,
                      is_composed=is_composed, result_path=path_to_save)


def run_class_kc2_case(sa_class: str, is_composed: bool = False, path_to_save=None):
    train_data, test_data = get_kc2_data()
    task = Task(TaskTypesEnum.classification)
    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    if is_composed:
        case = f'kc2_composed_{sa_class}'
        is_composed = True
    else:
        case = f'kc2_{sa_class}'
        is_composed = False

    run_case_analysis(train_data=train_data,
                      test_data=test_data,
                      case_name=case, task=task, sa_class=sa_class,
                      metric=metric_function,
                      is_composed=is_composed, result_path=path_to_save)


def run_regr_case(sa_class: str, is_composed: bool = False, path_to_save=None):
    train_data, test_data = get_cholesterol_data()
    task = Task(TaskTypesEnum.regression)
    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    if is_composed:
        case = f'cholesterol_composed_{sa_class}'
        is_composed = True
    else:
        case = f'cholesterol_{sa_class}'
        is_composed = False

    run_case_analysis(train_data=train_data,
                      test_data=test_data,
                      case_name=case, task=task, sa_class=sa_class,
                      metric=metric_function,
                      is_composed=is_composed, result_path=path_to_save)


if __name__ == '__main__':
    # You can assign any of PipelineSensitivityAnalysis, NodesAnalysis, PipelineAnalysis to any case

    # scoring case
    run_class_scoring_case(is_composed=False, sa_class='PipelineSensitivityAnalysis')

    # kc2 case
    run_class_kc2_case(is_composed=False, sa_class='NodesAnalysis')

    # cholesterol case
    run_regr_case(is_composed=False, sa_class='PipelineAnalysis')
