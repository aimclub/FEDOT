from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.benchmark_utils import get_cancer_case_data_paths, get_models_hyperparameters, save_metrics_result_file
from benchmark.executor import CaseExecutor, ExecutionParams
from core.repository.tasks import TaskTypesEnum

if __name__ == '__main__':
    train_file, test_file = get_cancer_case_data_paths()

    result_metrics = CaseExecutor(params=ExecutionParams(train_file=train_file,
                                                         test_file=test_file,
                                                         task=TaskTypesEnum.classification,
                                                         case_label='cancer',
                                                         target_name='target'),
                                  models=[BenchmarkModelTypesEnum.tpot,
                                          BenchmarkModelTypesEnum.h2o,
                                          BenchmarkModelTypesEnum.autokeras,
                                          BenchmarkModelTypesEnum.mlbox,
                                          BenchmarkModelTypesEnum.baseline],
                                  metric_list=['roc_auc', 'f1']).execute()

    result_metrics['hyperparameters'] = get_models_hyperparameters()

    save_metrics_result_file(result_metrics, file_name='cancer_metrics')
