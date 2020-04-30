from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.benchmark_utils import get_cancer_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor
from core.repository.task_types import MachineLearningTasksEnum

if __name__ == '__main__':
    train_file, test_file = get_cancer_case_data_paths()

    result_metrics = CaseExecutor(train_file=train_file,
                                  test_file=test_file,
                                  task=MachineLearningTasksEnum.classification,
                                  case_label='cancer',
                                  target_name='target',
                                  models=[BenchmarkModelTypesEnum.tpot,
                                          BenchmarkModelTypesEnum.h2o,
                                          BenchmarkModelTypesEnum.autokeras,
                                          BenchmarkModelTypesEnum.mlbox,
                                          BenchmarkModelTypesEnum.baseline]).execute()

    result_metrics['hyperparameters'] = get_models_hyperparameters()

    save_metrics_result_file(result_metrics, file_name='cancer_metrics')
