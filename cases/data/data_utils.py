import os
from typing import Tuple

from fedot.core.utils import project_root


def get_scoring_case_data_paths() -> Tuple[str, str]:
    train_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_train.csv')
    test_file_path = os.path.join('cases', 'data', 'scoring', 'scoring_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path
