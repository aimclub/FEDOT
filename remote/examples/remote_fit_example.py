import os
import random
from datetime import datetime

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from remote.remote_fit import ComputationalSetup

random.seed(1)
np.random.seed(1)

num_parallel = 3  # NUMBER OF PARALLEL TASKS

# LOCAL RUN

path = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring', 'scoring_train.csv')

start = datetime.now()
data = InputData.from_csv(path)
data.subset_list([1, 2, 3, 4, 5, 20])
pipeline = Pipeline(PrimaryNode('xgboost'))
pipeline.fit_from_scratch(data)
end = datetime.now()

print('LOCAL EXECUTION TIME', end - start)

# REMOTE RUN

ComputationalSetup.remote_eval_params = {
    'mode': 'remote',
    'dataset_name': 'scoring',
    'task_type': 'Task(TaskTypesEnum.classification)',
    'train_data_idx': [1, 2, 3, 4, 5, 20],
    'max_parallel': num_parallel,  # NUMBER OF PARALLEL TASKS
}

pipelines = [Pipeline(PrimaryNode('xgboost'))] * num_parallel
ComputationalSetup().fit(pipelines)
