import os
import random
from datetime import datetime

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from infrastructure.remote_fit import RemoteFitter

random.seed(1)
np.random.seed(1)

num_parallel = 3  # NUMBER OF PARALLEL TASKS

# LOCAL RUN

path = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring', 'scoring_comp.csv')

start = datetime.now()
data = InputData.from_csv(path)
pipeline = Pipeline(PrimaryNode('xgboost'))
pipeline.fit(data)
end = datetime.now()

print('LOCAL EXECUTION TIME', end - start)

# REMOTE RUN

RemoteFitter.remote_eval_params = {
    'use': True,
    'dataset_name': 'scoring',
    'task_type': 'Task(TaskTypesEnum.classification)',
    'test_sampele_idx': 15000,
    'max_parallel': num_parallel,  # NUMBER OF PARALLEL TASKS
}

pipelines = [Pipeline(PrimaryNode('xgboost'))] * num_parallel
RemoteFitter().fit(pipelines)
