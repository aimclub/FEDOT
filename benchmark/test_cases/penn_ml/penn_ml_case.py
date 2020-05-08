from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_penn_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor
from core.repository.task_types import MachineLearningTasksEnum
from pmlb import classification_dataset_names, regression_dataset_names
from pmlb import fetch_data

if __name__ == '__main__':
    with open('./datasets.txt', 'r') as f:
        datasets = f.read().splitlines()
    print(datasets)
    if len(datasets) == 0:
        datasets=classification_dataset_names+regression_dataset_names

    adult_data = fetch_data('adult')
    print(adult_data.describe())

    for name_of_dataset in datasets:
        if name_of_dataset in classification_dataset_names:
            problem_class=MachineLearningTasksEnum.classification
        else:
            problem_class=MachineLearningTasksEnum.regression
        print(name_of_dataset)
        train_file,test_file=get_penn_case_data_paths(name_of_dataset)
        config_models_data=get_models_hyperparameters()
        case_name='penn_ml'+str(name_of_dataset)

        
        result_metrics=CaseExecutor(train_file=train_file,
                                      test_file=test_file,
                                      task=problem_class,
                                      models=[ModelTypesEnum.tpot,
                                              ModelTypesEnum.h2o,
                                              ModelTypesEnum.fedot,
                                              ModelTypesEnum.autokeras,
                                              ModelTypesEnum.mlbox],
                                      target_name='target',
                                      case_label=case_name,hyperparameters=config_models_data).execute()

        result_metrics['hyperparameters']=config_models_data

        save_metrics_result_file(result_metrics,file_name='penn_ml_metrics_for '+str(name_of_dataset))
