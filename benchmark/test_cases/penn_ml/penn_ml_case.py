from pathlib import Path

import pandas as pd
from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from pmlb.write_metadata import imbalance_metrics

from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.benchmark_utils import convert_json_stats_to_csv, \
    get_models_hyperparameters, get_penn_case_data_paths, \
    save_metrics_result_file
from benchmark.executor import CaseExecutor, ExecutionParams
from core.repository.tasks import TaskTypesEnum


def _problem_and_metric_for_dataset(name_of_dataset: str, num_classes: int):
    if num_classes == 2 and name_of_dataset in classification_dataset_names:
        return TaskTypesEnum.classification, ['roc_auc', 'f1']
    elif name_of_dataset in regression_dataset_names:
        return TaskTypesEnum.regression, ['mse', 'r2']
    else:
        return None, None


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
        problem_class, metric_names = _problem_and_metric_for_dataset(name_of_dataset, num_classes)
        if not problem_class or not metric_names:
            print('Incorrect dataset')
            continue

        train_file, test_file = get_penn_case_data_paths(name_of_dataset)
        config_models_data = get_models_hyperparameters()
        case_name = f'penn_ml_{name_of_dataset}'

        try:
            result_metrics = CaseExecutor(params=ExecutionParams(train_file=train_file,
                                                                 test_file=test_file,
                                                                 task=problem_class,
                                                                 target_name='target',
                                                                 case_label=case_name),
                                          models=[BenchmarkModelTypesEnum.tpot,
                                                  BenchmarkModelTypesEnum.baseline,
                                                  BenchmarkModelTypesEnum.fedot],
                                          metric_list=metric_names).execute()
        except ValueError as ex:
            print(f'problems_with_dataset_{name_of_dataset}: {ex}')
            continue

        result_metrics['hyperparameters'] = config_models_data

        save_metrics_result_file(result_metrics, file_name=f'penn_ml_metrics_for_{name_of_dataset}')

    convert_json_stats_to_csv(dataset)
