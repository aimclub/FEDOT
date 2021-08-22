import os

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

data = InputData.from_csv(file_path=
                          os.path.join(fedot_project_root(), 'cases', 'data', 'oil.csv'),
                          task=Task(TaskTypesEnum.regression),
                          data_type=DataTypesEnum.table, delimiter=',')
data2 = InputData.from_csv(file_path=
                           os.path.join(fedot_project_root(), 'cases', 'data', 'oil2.csv'),
                           task=Task(TaskTypesEnum.regression),
                           data_type=DataTypesEnum.table, delimiter=',')
pipeline = Pipeline(PrimaryNode('linear'))
pipeline.fit(data)
pipeline.predict(data2)
