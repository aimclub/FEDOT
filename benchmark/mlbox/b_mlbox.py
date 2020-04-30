import os

import pandas as pd
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor
from mlbox.preprocessing import Reader, Drift_thresholder
from sklearn.metrics import roc_auc_score

from benchmark.benchmark_utils import get_scoring_case_data_paths, get_models_hyperparameters
from core.repository.task_types import MachineLearningTasksEnum


def separate_target_column(file_path: str, target_name: str):
    df = pd.read_csv(file_path)
    target = df[target_name].values

    df = df.drop([target_name], axis=1)

    path_to_file, _ = os.path.split(file_path)
    new_filename = 'm_cancer_test.csv'
    new_file_path = os.path.join(path_to_file, new_filename)

    df.to_csv(new_file_path, index=False)

    return new_file_path, target


def run_mlbox(train_file_path: str, test_file_path: str, target_name: str,
              task: MachineLearningTasksEnum):
    config_data = get_models_hyperparameters()['MLBox']
    new_test_file_path, test_target = separate_target_column(test_file_path, target_name)
    paths = [train_file_path, new_test_file_path]

    data = Reader(sep=",").train_test_split(paths, target_name)
    data = Drift_thresholder().fit_transform(data)

    score = 'roc_auc' if task is MachineLearningTasksEnum.classification else 'neg_mean_squared_error'

    opt = Optimiser(scoring=score, n_folds=5)
    params = opt.optimise(config_data['space'], data, max_evals=config_data['max_evals'])
    opt.evaluate(params, data)

    Predictor(verbose=False).fit_predict(params, data)

    cur_work_dir = os.path.abspath(os.curdir)

    predicted_df = pd.read_csv(os.path.join(cur_work_dir, f'save/{target_name}_predictions.csv'))
    predicted = predicted_df['1.0']
    metric = roc_auc_score(test_target, predicted)

    print(f'ROC_AUC: {metric}')

    os.remove(new_test_file_path)

    return metric


if __name__ == '__main__':
    train_path, test_path = get_scoring_case_data_paths()
    run_mlbox(train_file_path=train_path,
              test_file_path=test_path,
              target_name='default',
              task=MachineLearningTasksEnum.classification)
