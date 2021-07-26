import pandas as pd
import numpy as np
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

len_forecast = 40
ts_name = 'sea_level'
path_to_file = 'cases/data/nemo/sea_surface_height.csv'

df = pd.read_csv(path_to_file)
time_series = np.array(df[ts_name])

# Let's divide our data on train and test samples
train_data = time_series[:-len_forecast]
test_data = time_series[-len_forecast:]

task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

start_forecast = len(train_data)
end_forecast = start_forecast + len_forecast
predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)


node_lagged = PrimaryNode('lagged')
node_lagged.custom_params = {'window_size': 50}
node_root = SecondaryNode('ridge', nodes_from=[node_lagged])

pipeline = Pipeline(node_root)
pipeline.fit(train_input)
predicted_values = pipeline.predict(predict_input).predict


