from pathlib import Path

import pandas as pd
from pmlb import classification_dataset_names, regression_dataset_names, fetch_data
from pmlb.write_metadata import imbalance_metrics

from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.benchmark_utils import get_penn_case_data_paths, save_metrics_result_file, get_models_hyperparameters
from benchmark.executor import CaseExecutor
from core.repository.task_types import MachineLearningTasksEnum

if __name__ == '__main__':
    penn_data = Path('./datasets.csv')
    dataset = []
    if penn_data.is_file():
        df = pd.read_csv(penn_data)
        dataset = df['dataset_names'].values
    else:
        print('Please create nonempty csv-file with datasets')

    if len(dataset) == 0:
        dataset = classification_dataset_names + regression_dataset_names

    for name_of_dataset in dataset:
        pmlb_data = fetch_data(name_of_dataset)
        num_classes, _ = imbalance_metrics(pmlb_data['target'].tolist())
        if num_classes == 2 and name_of_dataset in classification_dataset_names:
            problem_class = MachineLearningTasksEnum.classification
            metric_name = ['roc_auc', 'f1']
        elif name_of_dataset in regression_dataset_names:
            problem_class = MachineLearningTasksEnum.regression
            metric_name = ['mse', 'r2']
        else:
            print('Incorrect dataset')
            continue

        train_file, test_file = get_penn_case_data_paths(name_of_dataset)
        config_models_data = get_models_hyperparameters()
        case_name = f'penn_ml_{name_of_dataset}'

        result_metrics = CaseExecutor(train_file=train_file,
                                      test_file=test_file,
                                      task=problem_class,
                                      models=[BenchmarkModelTypesEnum.tpot,
                                              BenchmarkModelTypesEnum.baseline,
                                              BenchmarkModelTypesEnum.fedot],
                                      target_name='target',
                                      case_label=case_name,
                                      metric_list=metric_name).execute()

        result_metrics['hyperparameters'] = config_models_data

        save_metrics_result_file(result_metrics, file_name=f'penn_ml_metrics_for_{name_of_dataset}')
