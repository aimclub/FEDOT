import numpy as np
import pandas as pd

from examples.advanced.pipelines_caching import PreprocessingCacheMock
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.preprocessing.cache import PreprocessingCache


def generate_gaps_and_categories(input_data: InputData):
    """ Generate gaps and categories in the dataset """
    gap_percentage = 20
    features = np.array(input_data.features, dtype=object)

    # Generate dummy column with categorical values
    new_categorical_column = np.array(['a'] * len(features))
    random_indices = list(set(np.random.random_integers(0, len(new_categorical_column) - 1, 5000)))
    new_categorical_column[random_indices] = 'b'
    random_indices = list(set(np.random.random_integers(0, len(new_categorical_column) - 1, 5000)))
    new_categorical_column[random_indices] = 'c'

    # Generate gaps
    n_rows, n_cols = features.shape
    frames = [new_categorical_column.reshape(-1, 1)]
    for column_id in range(n_cols):
        current_column = features[:, column_id]

        number_of_gaps = round(len(current_column) * (gap_percentage / 100))
        random_indices = list(set(np.random.random_integers(0, len(current_column) - 1, number_of_gaps)))
        current_column[random_indices] = np.nan
        frames.append(current_column.reshape(-1, 1))

    input_data.features = np.hstack(frames)

    return input_data


def run_example_with(train, test, problem: str):
    if os.getenv('use_preproc_caching') == '':
        PreprocessingCache.add_preprocessor.__code__ = PreprocessingCacheMock.add_preprocessor.__code__
        PreprocessingCache.try_find_preprocessor.__code__ = PreprocessingCacheMock.try_find_preprocessor.__code__

    composer_params = {'history_folder': 'custom_history_folder', 'cv_folds': None}
    auto_model = Fedot(problem=problem, seed=42, composer_params=composer_params,
                       preset='auto',
                       timeout=3,
                       verbose_level=0,
                       use_default_preprocessors=True)

    auto_model.fit(features=train, target='target')
    if auto_model.history is not None:
        auto_model.history.save('saved_regression_history.json')
    prediction = auto_model.predict(features=test)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


if __name__ == '__main__':
    import os

    os.environ['preproc_caching_save_pth'] = 'with_cache_clean'
    os.environ['use_preproc_caching'] = '1'
    problem = 'classification'

    if problem == 'classification':
        train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
        test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)
        cat_columns = [
            'NumberOfTime30.59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines', 'NumberOfTime60.89DaysPastDueNotWorse', 'NumberOfDependents']
        train = train.drop(columns=cat_columns).dropna()
        test = test.drop(columns=cat_columns).dropna()
    elif problem == 'regression':
        data_path = f'{fedot_project_root()}/cases/data/cal_housing.csv'

        data = InputData.from_csv(data_path,
                                  index_col=None,
                                  task=Task(TaskTypesEnum.regression),
                                  target_columns='medianHouseValue')
        data = generate_gaps_and_categories(data)
        train, test = train_test_data_setup(data)
    run_example_with(train, test, problem)
