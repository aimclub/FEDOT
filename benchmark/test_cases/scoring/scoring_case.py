from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_scoring_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor
from core.repository.task_types import MachineLearningTasksEnum

if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    config_models_data = get_models_hyperparameters()

    result_metrics = CaseExecutor(train_file=train_file,
                                  test_file=test_file,
                                  task=MachineLearningTasksEnum.classification,
                                  models=[ModelTypesEnum.tpot,
                                          ModelTypesEnum.h2o,
                                          ModelTypesEnum.fedot,
                                          ModelTypesEnum.autokeras,
                                          ModelTypesEnum.mlbox],
                                  target_name='default',
                                  case_label='scoring', hyperparameters=config_models_data).execute()

    result_metrics['hyperparameters'] = config_models_data

    save_metrics_result_file(result_metrics, file_name='scoring_metrics')
