import os

from core.utils import project_root
import json


def get_scoring_case_data_paths():
    train_file_path = 'cases/data/scoring/scoring_train.csv'
    test_file_path = 'cases/data/scoring/scoring_test.csv'
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def save_metrics_result_file(data: dict, file_name: str):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(data, file, indent=4)
