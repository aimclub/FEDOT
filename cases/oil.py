import os

import pandas as pd
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root, train_test_split


data = pd.read_csv(os.path.join(fedot_project_root(), 'cases', 'data', 'oil.csv'))
data2 = pd.read_csv(os.path.join(fedot_project_root(), 'cases', 'data', 'oil.csv'))
full_data = pd.concat([data, data2], ignore_index=True).drop(columns=['Index '], axis=1).reset_index()
full_target = full_data['target ']
full_data.drop(columns=['target '], axis=1, inplace=True)
train, test, train_target, test_target = train_test_split(full_data, full_target)

data = InputData(features=np.array(train), task=Task(TaskTypesEnum.regression),
                 data_type=DataTypesEnum.table, idx=np.arange(len(train)), target=np.array(train_target))

data2 = InputData(features=np.array(test), task=Task(TaskTypesEnum.regression),
                  data_type=DataTypesEnum.table, idx=np.arange(len(test)), target=np.array(test_target))

pipeline = Pipeline(PrimaryNode('linear'))
pipeline.fit(data)
pipeline.predict(data2)
