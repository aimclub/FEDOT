# This script generates train ts, test ts and fedot model for testing prediction intervals

import numpy as np
import matplotlib.pyplot as plt
import pickle

from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.data.data import InputData
from fedot.api.main import Fedot
from fedot.core.repository.dataset_types import DataTypesEnum


def synthetic_series(start, end):

    trend = np.array([5 * np.sin(x / 20) + 0.1 * x - 2 * np.sqrt(x) for x in range(start, end)])
    noise = np.random.normal(loc=0, scale=1, size=end - start)

    return trend + noise


ts_train = synthetic_series(0, 200)
ts_test = synthetic_series(200, 220)
np.savetxt("train_ts.csv", ts_train, delimiter=",")
np.savetxt("test_ts.csv", ts_test, delimiter=",")

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

with open('prediction_intervals_fedot_model.pickle', 'wb') as f:
    pickle.dump(model, f)
