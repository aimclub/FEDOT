import os
import random

import numpy as np

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

random.seed(1)
np.random.seed(1)

fitter = RemoteEvaluator(RemoteTaskParams(
    mode='remote',
    dataset_name='scoring',
    task_type='Task(TaskTypesEnum.classification)',
    train_data_idx=None,
    is_multi_modal=False,
    var_names=None,
    max_parallel=2))

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
