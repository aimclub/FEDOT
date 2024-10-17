import os
from typing import Tuple

from fedot.core.utils import fedot_project_root


def get_scoring_case_data_paths() -> Tuple[str, str]:
    train_file_path = os.path.join('examples', 'real_cases', 'data', 'scoring', 'scoring_train.csv')
    test_file_path = os.path.join('examples', 'real_cases', 'data', 'scoring', 'scoring_test.csv')
    full_train_file_path = os.path.join(str(fedot_project_root()), train_file_path)
    full_test_file_path = os.path.join(str(fedot_project_root()), test_file_path)

    return full_train_file_path, full_test_file_path
