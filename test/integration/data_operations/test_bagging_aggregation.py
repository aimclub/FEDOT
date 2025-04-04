import numpy as np
from sklearn.metrics import accuracy_score as acc
from sklearn.datasets import load_iris
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data_split import train_test_data_setup


def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def test_bagging_aggregation():
    data = data_setup()
    train, test = train_test_data_setup(data)
    pipeline = PipelineBuilder().add_node('scaling').add_branch('catboost', 'xgboost', 'lgbm').join_branches(
        'bagging').build()

    pipeline.fit(train)
    output = pipeline.predict(test)
    y_true, y_pred = output.target, output.predict
    score = round(acc(y_true, y_pred))

    classes_count = 3  # for iris dataset

    assert output.features.shape[1] == classes_count
    assert score > 0.5
