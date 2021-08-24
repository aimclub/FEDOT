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
    'dataset_name': 'cholesterol',
    'task_type': 'Task(TaskTypesEnum.regression)'
}

preset = 'light_tun'
automl = Fedot(problem='regression', timeout=2, verbose_level=4,
               preset=preset)
path = os.path.join(fedot_project_root(), 'cases', 'data', 'cholesterol', 'cholesterol.csv')

automl.fit(path)
predict = automl.predict(path)
metrics = automl.get_metrics()

automl.current_pipeline.show()
