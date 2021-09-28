import pandas as pd
import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository import tasks
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root
from fedot.utilities.ts_gapfilling import ModelGapFiller

forecast_len = 2
original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
gap_value = -100


def filling_gaps(arr):
    ids_with_gaps = [12, 13, 14, 15]
    arr[ids_with_gaps] = -100

    return arr


task_parameters = tasks.TsForecastingParams(forecast_length=forecast_len)
task = tasks.Task(tasks.TaskTypesEnum.ts_forecasting, task_parameters)

model = Fedot(
    problem='ts_forecasting',
    task_params=task_parameters,
    composer_params={'timeout': 0.1})

_input = InputData(idx=np.arange(0, len(original_array)),
                   features=original_array,
                   target=original_array,
                   task=task, data_type=DataTypesEnum.ts)

pipeline = model.fit(_input)
# Correct window_size parameters
for i, node in enumerate(pipeline.nodes):
    current_operation = node.operation.operation_type
    if current_operation == 'lagged' or current_operation == 'sparse_lagged':
        pipeline.nodes[i].custom_params = {'window_size': 2}

gapfiller = ModelGapFiller(gap_value=gap_value,
                           pipeline=pipeline)

array_with_gaps = filling_gaps(original_array) # a function to add gap_value
output = gapfiller.forward_inverse_filling(array_with_gaps)