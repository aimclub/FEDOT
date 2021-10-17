import os
import random
from datetime import datetime

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

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

remote_eval_params = RemoteTaskParams(
    mode='remote',
    dataset_name='scoring_train',
    task_type='Task(TaskTypesEnum.classification)',
    train_data_idx=None,
    max_parallel=num_parallel
)

pipelines = [Pipeline(PrimaryNode('xgboost'))] * num_parallel
setup = RemoteEvaluator(remote_eval_params)
setup.compute_pipelines(pipelines)

[print(p.is_fitted) for p in pipelines]
