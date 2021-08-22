import os

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

data = InputData.from_csv(file_path=
                          os.path.join(fedot_project_root(), 'cases', 'data', 'scoring', 'scoring_train.csv'),
                          task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.table)
pipeline = Pipeline(PrimaryNode('lda'))
pipeline.fit(data)
