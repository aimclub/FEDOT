import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def get_data_from_csv(data_path: Path, task_type: TaskTypesEnum, shuffle_flag: bool = True):
    data_frame = pd.read_csv(data_path)
    dataset_name = os.path.split(data_path)[-1]
    if dataset_name in ['sylvine.csv', 'jasmine.csv', 'volkert.csv']:
        targets = data_frame[data_frame.columns[0]].values
        features = data_frame.drop([f'{data_frame.columns[0]}'], axis=1).values
    else:
        targets = data_frame[data_frame.columns[-1]].values
        features = data_frame.drop([f'{data_frame.columns[-1]}'], axis=1).values

    data = InputData(features=features, target=targets, idx=np.arange(0, len(targets)),
                     task=Task(task_type),
                     data_type=DataTypesEnum.table
                     )
    train_data, test_data = train_test_data_setup(data, split_ratio=0.7, shuffle_flag=shuffle_flag)

    return train_data, test_data


def run_classification_example(timeout: float = None):
    # train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    # test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'
    #
    problem = 'classification'

    train_data, test_data = get_data_from_csv(Path(f'{fedot_project_root()}/examples/data/kc1.csv'),
                                              task_type=TaskTypesEnum.classification)

    baseline_model = Fedot(problem=problem, timeout=timeout)
    baseline_model.fit(features=train_data, target='target', predefined_model='rf')

    baseline_model.predict(features=test_data)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, seed=42, timeout=timeout, n_jobs=-1,
                       max_pipeline_fit_time=1, composer_metric='roc_auc', tuner_metric='roc_auc')
    auto_model.fit(features=train_data, target='target')
    prediction = auto_model.predict_proba(features=test_data)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()
    return prediction


if __name__ == '__main__':
    run_classification_example(timeout=2)
