import os

from core.utils import project_root


def get_initial_data_paths():
    train_file_path = 'cases/data/scoring/scoring_train.csv'
    test_file_path = 'cases/data/scoring/scoring_test.csv'
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path
