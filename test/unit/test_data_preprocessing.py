import numpy as np
import pytest

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def data_with_only_categorical_features():
    """ Generate tabular data with only categorical features. All of them are binary. """
    task = Task(TaskTypesEnum.regression)
    features = np.array([['a', '0', '1'],
                         ['b', '1', '0'],
                         ['c', '1', '0']], dtype=object)
    input = InputData(idx=[0, 1, 2], features=features,
                      target=np.array([0, 1, 2]),
                      task=task,  data_type=DataTypesEnum.table)
    return input


def data_with_too_much_nans():
    """ Generate tabular data with too much nan's in numpy array (inf values also must be signed as nan).
    Columns with ids 1 and 2 have nans more than 30% in their structure.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, np.inf, np.nan],
                         [np.nan, np.inf, np.nan],
                         [3, np.inf, np.nan],
                         [7, np.inf, np.nan],
                         [8, '1', np.nan],
                         [np.nan, '0', 23],
                         [9, np.inf, 22],
                         [9, np.inf, np.nan],
                         [9, np.inf, np.nan],
                         [9, '1', np.inf]], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table)

    return train_input


def data_with_spaces_and_nans_in_features():
    """
    Generate InputData with categorical features with leading and
    trailing spaces. Dataset contains np.nan also.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([['1 ', '1 '],
                         [np.nan, ' 0'],
                         [' 1 ', np.nan],
                         ['1 ', '0  '],
                         ['0  ', '  1'],
                         ['1 ', '  0']], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table)

    return train_input


def data_with_nans_in_target_column():
    """ Generate InputData with np.nan values in target column """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]], dtype=object)
    target = np.array([[0], [1], [np.nan], [np.nan], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table)

    return train_input


def data_with_nans_in_multi_target():
    """
    Generate InputData with np.nan values in target columns.
    So the multi-output regression task is solved.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]], dtype=object)
    target = np.array([[0, 2], [1, 3], [np.nan, np.nan], [3, np.nan], [4, 4], [5, 6]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table)

    return train_input


def test_correct_api_dataset_preprocessing():
    """ Check if dataset preprocessing was performed correctly when API launch using. """
    funcs = [data_with_only_categorical_features, data_with_too_much_nans,
             data_with_spaces_and_nans_in_features, data_with_nans_in_target_column,
             data_with_nans_in_multi_target]

    # Check for all datasets
    for data_generator in funcs:
        input_data = data_generator()
        fedot_model = Fedot(problem='regression', check_mode=True)
        with pytest.raises(SystemExit) as exc:
            assert fedot_model.fit(input_data)
        assert str(exc.value) == f'Initial pipeline were fitted successfully'
