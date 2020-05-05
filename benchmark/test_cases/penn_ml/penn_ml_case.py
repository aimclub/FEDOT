from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_penn_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor
from core.repository.task_types import MachineLearningTasksEnum

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise TypeError('Please enter a type of ML_problem as argument.You can choose: classification or regression')
    type_flag = str(sys.argv[1])
    train_file, test_file = get_penn_case_data_paths(type_flag)
    config_models_data = get_models_hyperparameters()
    if type_flag == 'classification':
        problem_class = MachineLearningTasksEnum.classification
    else:
        problem_class = MachineLearningTasksEnum.regression
        
    result_metrics = CaseExecutor(train_file=train_file,
                                  test_file = test_file,
                                  task = problem_class,
                                  models=[ModelTypesEnum.tpot,
                                          ModelTypesEnum.h2o,
                                          ModelTypesEnum.fedot,
                                          ModelTypesEnum.autokeras,
                                          ModelTypesEnum.mlbox],
                                  target_name='target',
                                  case_label='penn_ml', hyperparameters=config_models_data).execute()

    result_metrics['hyperparameters'] = config_models_data

    save_metrics_result_file(result_metrics, file_name='penn_ml_metrics')
