import numpy as np
import pandas as pd
import pytest

from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


def get_dataset_with_cats(output_mode: str = None):
    path_to_csv = fedot_project_root().joinpath('test/data/melb_data.csv')
    df = pd.read_csv(path_to_csv)

    if output_mode == 'path':
        return path_to_csv, 'Price'

    elif output_mode == 'dataframe':
        return df.drop(['Price'], axis=1), df['Price']

    elif output_mode == 'numpy':
        return df.drop(['Price'], axis=1).to_numpy(), df.Price.to_numpy(), df.columns.values


def get_dataset_without_cats(output_mode: str = None):
    path_to_csv = fedot_project_root().joinpath('test/data/scoring/scoring_train.csv')
    df = pd.read_csv(path_to_csv)
    df = df.drop(['ID'], axis=1)

    if output_mode == 'path':
        return path_to_csv, 'target'

    elif output_mode == 'dataframe':
        return df.drop(['target'], axis=1), df['target']

    elif output_mode == 'numpy':
        return df.drop(['target'], axis=1).to_numpy(), df.target.to_numpy(), df.columns.values


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([0, 1, 2, 3, 6, 7])),
    ([], np.array([]), np.array([0, 1, 2])),
    (np.array([]), np.array([]), np.array([0, 1, 2])),
    (['Type', 'Method', 'Regionname'], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array(['Type', 'Method', 'Regionname']), np.array([0, 1, 2]), np.array([0, 1, 2])),
    ([0, 1, 2], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2]))
])
def test_from_numpy_with_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    X, y, features_names = get_dataset_with_cats(output_mode='numpy')

    input_data = InputData.from_numpy(
        features_array=X,
        target_array=y,
        features_names=features_names,
        categorical_idx=categorical_idx,
        task='regression'
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([0, 1, 2, 3, 6, 7])),
    ([], np.array([]), np.array([0, 1, 2])),
    (np.array([]), np.array([]), np.array([0, 1, 2])),
    (['Type', 'Method', 'Regionname'], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array(['Type', 'Method', 'Regionname']), np.array([0, 1, 2]), np.array([0, 1, 2])),
    ([0, 1, 2], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2]))
])
def test_from_dataframe_with_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    X_df, y_df = get_dataset_with_cats(output_mode='dataframe')

    input_data = InputData.from_dataframe(
        features_df=X_df,
        target_df=y_df,
        categorical_idx=categorical_idx,
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([0, 1, 2, 3, 6, 7])),
    ([], np.array([]), np.array([0, 1, 2])),
    (np.array([]), np.array([]), np.array([0, 1, 2])),
    (['Type', 'Method', 'Regionname'], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array(['Type', 'Method', 'Regionname']), np.array([0, 1, 2]), np.array([0, 1, 2])),
    ([0, 1, 2], np.array([0, 1, 2]), np.array([0, 1, 2])),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2]))
])
def test_from_csv_with_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    path, target_columns = get_dataset_with_cats(output_mode='path')

    input_data = InputData.from_csv(
        file_path=path,
        target_columns=target_columns,
        categorical_idx=categorical_idx
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([2, 6, 7, 9])),
    ([], np.array([]), np.array([])),
    (np.array([]), np.array([]), np.array([])),
])
def test_from_numpy_without_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    X, y, features_names = get_dataset_without_cats(output_mode='numpy')

    input_data = InputData.from_numpy(
        features_array=X,
        target_array=y,
        features_names=features_names,
        categorical_idx=categorical_idx,
        task='regression'
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([2, 6, 7, 9])),
    ([], np.array([]), np.array([])),
    (np.array([]), np.array([]), np.array([])),
])
def test_from_dataframe_without_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    X_df, y_df = get_dataset_without_cats(output_mode='dataframe')

    input_data = InputData.from_dataframe(
        features_df=X_df,
        target_df=y_df,
        categorical_idx=categorical_idx,
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()


@pytest.mark.parametrize('categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing', [
    (None, None, np.array([2, 6, 7, 9])),
    ([], np.array([]), np.array([])),
    (np.array([]), np.array([]), np.array([])),
])
def test_from_csv_without_cats(categorical_idx, expected_idx_after_opening, expected_idx_after_preprocessing):
    path, target_columns = get_dataset_without_cats(output_mode='path')

    input_data = InputData.from_csv(
        file_path=path,
        target_columns=target_columns,
        categorical_idx=categorical_idx
    )

    if isinstance(input_data.categorical_idx, np.ndarray):
        assert (input_data.categorical_idx == expected_idx_after_opening).all()
    else:
        assert input_data.categorical_idx == expected_idx_after_opening

    data_preprocessor = ApiDataProcessor(task=Task(TaskTypesEnum.classification))
    preprocessed_input_data = data_preprocessor.fit_transform(input_data)

    assert (preprocessed_input_data.categorical_idx == expected_idx_after_preprocessing).all()
