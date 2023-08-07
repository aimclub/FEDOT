import os
import random
from datetime import datetime

import numpy as np

from examples.simple.classification.classification_pipelines import classification_three_depth_manual_pipeline
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.datamall_client import DataMallClient, DEFAULT_CONNECT_PARAMS, \
    DEFAULT_EXEC_PARAMS
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams

random.seed(1)
np.random.seed(1)

num_parallel = 10  # NUMBER OF PARALLEL TASKS

# WARNING - THIS SCRIPT CAN BE EVALUATED ONLY WITH THE ACCESS TO DATAMALL SYSTEM

# LOCAL RUN
folder = os.path.join(fedot_project_root(), 'cases', 'data', 'scoring')
path = os.path.join(folder, 'scoring_train.csv')

model = 'rf'



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

#  following client and params can be used with DataMall system
connect_params = DEFAULT_CONNECT_PARAMS
exec_params = DEFAULT_EXEC_PARAMS
client = DataMallClient(connect_params, exec_params, output_path=os.path.join(folder, 'remote'))

evaluator = RemoteEvaluator()
evaluator.init(
    client=client,
    remote_task_params=remote_task_params
)

start = datetime.now()

pipelines = [classification_three_depth_manual_pipeline()] * num_parallel
fitted_pipelines = evaluator.compute_graphs(pipelines)

[print(p.is_fitted) for p in fitted_pipelines]

end = datetime.now()

print('REMOTE EXECUTION TIME', end - start)
