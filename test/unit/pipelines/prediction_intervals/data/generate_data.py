# This script generates fedot model for testing prediction intervals

import numpy as np
import matplotlib.pyplot as plt
import pickle

from numpy import genfromtxt

from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.data.data import InputData
from fedot import Fedot
from fedot.core.repository.dataset_types import DataTypesEnum


ts_train = genfromtxt('train_ts.csv', delimiter=',')
ts_test = genfromtxt('test_ts.csv', delimiter=',')

fig, ax = plt.subplots()
ax.plot(range(200), ts_train)
ax.plot(range(200, 220), ts_test)

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=20))
idx = np.array(range(len(np.array(ts_train))))
train_input = InputData(idx=idx,
                        features=ts_train,
                        target=ts_train,
                        task=task,
                        data_type=DataTypesEnum.ts)

model = Fedot(problem='ts_forecasting',
              task_params=task.task_params,
              timeout=3,
              preset='ts')
model.fit(train_input)
model.forecast()

with open('pred_ints_model_test.pickle', 'wb') as f:
    pickle.dump(model, f)
