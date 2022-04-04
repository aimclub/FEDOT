import pandas as pd
import numpy as np
import datetime

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def prepare_dataset(data_path, take_dataset_percent):
    df = pd.read_csv(data_path)
    all_cols = list(df.columns)

    if data_path == 'KDDCup09_appetency.csv':
        features_columns = all_cols[:-1]
        target_column = 'APPETENCY'
    else:
        features_columns = all_cols[:-1]
        target_column = 'label'

    if take_dataset_percent is not None and take_dataset_percent < 100:
        number_of_elements = round(len(df) * (take_dataset_percent/100))
        df = df.sample(n=number_of_elements,
                       random_state=1)
    print(f'Table size {df.shape}')

    features = np.array(df[features_columns])
    target = np.array(df[target_column])

    for unique_label in np.unique(target):
        number_of_labels = len(np.ravel(np.argwhere(target == unique_label)))
        label_percent = (number_of_labels / len(target)) * 100
        print(f'Percent of label {unique_label} in target is {label_percent:.1f}')

    input_data = InputData(idx=np.arange(0, len(target)),
                           features=features,
                           target=target,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def run_classification_example(timeout: float = None, take_dataset_percent: float = None):
    # Possible file names KDDCup09_appetency.csv, KDDCup99_full.csv
    input_data = prepare_dataset(f'KDDCup09_appetency.csv', take_dataset_percent)

    problem = 'classification'

    starting_time = datetime.datetime.now()

    auto_model = Fedot(problem=problem, seed=42, timeout=timeout, safe_mode=True)
    auto_model.fit(input_data, predefined_model='auto')

    spend_time = datetime.datetime.now() - starting_time
    print(f'Spend {spend_time} for fit method')
    prediction = auto_model.predict_proba(input_data)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


if __name__ == '__main__':
    run_classification_example(timeout=5, take_dataset_percent=100)
