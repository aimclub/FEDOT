import os
import random
from datetime import datetime

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.test_client import TestClient
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

random.seed(1)
np.random.seed(1)

num_parallel = 3  # NUMBER OF PARALLEL TASKS

# WARNING - THIS SCRIPT CAN BE EVALUATED ONLY WITH THE ACCESS TO DATAMALL SYSTEM

# LOCAL RUN
folder = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring')
path = os.path.join(folder, 'scoring_train.csv')

start = datetime.now()
data = InputData.from_csv(path)
data.subset_indices([1, 2, 3, 4, 5, 20])
pipeline = Pipeline(PrimaryNode('rf'))
pipeline.fit_from_scratch(data)
end = datetime.now()

print('LOCAL EXECUTION TIME', end - start)

# REMOTE RUN

connect_params = {}
exec_params = {
    'container_input_path': folder,
    'container_output_path': os.path.join(folder, 'remote'),
    'container_config_path': ".",
    'container_image': "test",
    'timeout': 1
}

remote_task_params = RemoteTaskParams(
    mode='remote',
    dataset_name='scoring_train',
    task_type='Task(TaskTypesEnum.classification)',
    max_parallel=num_parallel
)

client = TestClient(connect_params, exec_params, output_path=os.path.join(folder, 'remote'))

evaluator = RemoteEvaluator()
evaluator.init(
    client=client,
    remote_task_params=remote_task_params
)

pipelines = [Pipeline(PrimaryNode('rf'))] * num_parallel
fitted_pipelines = evaluator.compute_graphs(pipelines)

[print(p.is_fitted) for p in fitted_pipelines]
