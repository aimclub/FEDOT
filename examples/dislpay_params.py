import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository import tasks
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def get_pipeline():
    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_ridge)


if __name__ == '__main__':
    forecast_len = 2
    original_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    task_parameters = tasks.TsForecastingParams(forecast_length=forecast_len)
    task = tasks.Task(tasks.TaskTypesEnum.ts_forecasting, task_parameters)

    input_ts = InputData(idx=np.arange(0, len(original_array)),
                         features=original_array,
                         target=original_array,
                         task=task, data_type=DataTypesEnum.ts)

    # Get simple pipeline for time series forecasting
    ts_pipeline = get_pipeline()

    # Fit pipeline with inconceivably incorrect parameters (window_size will be corrected to 2)
    ts_pipeline.fit(input_ts)

    # Correct window_size parameter to new value
    ts_pipeline.nodes[1].custom_params = {'window_size': 3}

    ts_pipeline.print_structure()

    print(f'\ncustom_params: {ts_pipeline.nodes[1].custom_params}')
    print(f'content: {ts_pipeline.nodes[1].content}')
    print(f'descriptive_id: {ts_pipeline.nodes[1].descriptive_id}')
