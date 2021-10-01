import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task

if __name__ == '__main__':
    task = Task(TaskTypesEnum.classification)

    # train_input = InputData(
    #     idx=np.arange(0, 100),
    #     features=np.random.rand(100, 5),
    #     target=np.random.randint(0, 2, 100),
    #     task=task,
    #     data_type=DataTypesEnum.table
    # )

    train_input = InputData(
        idx=np.arange(0, 10),
        features=np.random.rand(10, 2),
        target=np.random.randint(0, 2, 10),
        task=task,
        data_type=DataTypesEnum.table,
    )

    """
    Pipeline(resample -> Logit)
    """
    node_resample = PrimaryNode(operation_type='resample')
    node_resample.custom_params = {
        "balance": "majority",
    }
    graph = SecondaryNode(operation_type='logit', nodes_from=[node_resample])

    pipeline = Pipeline(graph)

    pipeline.fit(train_input)
