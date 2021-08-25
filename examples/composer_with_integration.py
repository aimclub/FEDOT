import os
import random

import numpy as np

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root
from infrastructure.remote_fit import RemoteFitter

random.seed(1)
np.random.seed(1)

RemoteFitter.remote_eval_params = {
    'use': True,
    'dataset_name': 'scoring',
    'task_type': 'Task(TaskTypesEnum.classification)',
    'test_sampele_idx': 15000,
    'max_parallel': 5,
}

params = {
    'pop_size': 22,
    'cv_folds': None
}

preset = 'light'
automl = Fedot(problem='classification', timeout=2, verbose_level=4,
               preset=preset, composer_params=params)
path = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring', 'scoring_train.csv')
path_valid = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring', 'scoring_test.csv')

automl.fit(path)
predict = automl.predict(path_valid)
metrics = automl.get_metrics()

automl.current_pipeline.show()
