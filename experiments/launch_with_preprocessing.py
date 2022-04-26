import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


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
        random_indices = list(set(np.random.random_integers(0, len(current_column)-1, number_of_gaps)))
        current_column[random_indices] = np.nan
        frames.append(current_column.reshape(-1, 1))

    input_data.features = np.hstack(frames)

    return input_data


def run_regression_example():
    data_path = f'{fedot_project_root()}/cases/data/cal_housing.csv'

    data = InputData.from_csv(data_path,
                              index_col=None,
                              task=Task(TaskTypesEnum.regression),
                              target_columns='medianHouseValue')
    data = generate_gaps_and_categories(data)
    train, test = train_test_data_setup(data)
    problem = 'regression'

    composer_params = {'history_folder': 'custom_history_folder', 'cv_folds': None}
    auto_model = Fedot(problem=problem, seed=42, composer_params=composer_params,
                       preset='auto',
                       timeout=10,
                       verbose_level=1,
                       use_default_preprocessors=True)

    auto_model.fit(features=train, target='target')
    auto_model.history.save('saved_regression_history.json')
    prediction = auto_model.predict(features=test)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


if __name__ == '__main__':
    run_regression_example()
