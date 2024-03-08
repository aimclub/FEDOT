import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.text import SkLearnTextVectorizeStrategy
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_tags import ModelTagsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.pipelines.test_decompose_pipelines import get_classification_data
from test.unit.tasks.test_regression import get_synthetic_regression_data


def test_vectorize_tfidf_strategy():
    train_text = ['This document first' 'second This document' 'And one third'
                  'Is document first']
    test_text = ['document allow', 'spam not found', 'is are']

    train_data = InputData(idx=len(train_text), features=train_text,
                           target=[0, 0, 1, 0], data_type=DataTypesEnum.text,
                           task=Task(TaskTypesEnum.classification))
    test_data = InputData(idx=len(test_text), features=test_text,
                          target=[0, 1, 0], data_type=DataTypesEnum.text,
                          task=Task(TaskTypesEnum.classification))

    vectorizer = SkLearnTextVectorizeStrategy(operation_type='tfidf',
                                              params=None)

    vectorizer_fitted = vectorizer.fit(train_data)

    predicted = vectorizer.predict(trained_operation=vectorizer_fitted,
                                   predict_data=test_data)
    predicted_labels = predicted.predict

    assert isinstance(vectorizer_fitted, TfidfVectorizer)
    assert len(predicted_labels[0]) == 7


def test_boosting_classification_operation():
    train_data, test_data = get_classification_data()

    model_names = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.classification, tags=[ModelTagsEnum.boosting]
    )

    for model_name in model_names:
        pipeline = PipelineBuilder().add_node(model_name, params={'n_jobs': -1}).build()
        pipeline.fit(train_data)
        predicted_output = pipeline.predict(test_data, output_mode='labels')
        metric = roc_auc(test_data.target, predicted_output.predict)

        assert isinstance(pipeline, Pipeline)
        assert predicted_output.predict.shape[0] == 240
        assert metric > 0.5


def test_boosting_regression_operation():
    n_samples = 2000
    data = get_synthetic_regression_data(n_samples=n_samples, n_features=10, random_state=42)
    train_data, test_data = train_test_data_setup(data)

    model_names = OperationTypesRepository().suitable_operation(
        task_type=TaskTypesEnum.regression, tags=[ModelTagsEnum.boosting]
    )

    for model_name in model_names:
        pipeline = PipelineBuilder().add_node(model_name).build()
        pipeline.fit(train_data, n_jobs=-1)
        predicted_output = pipeline.predict(test_data)
        metric = mean_squared_error(test_data.target, predicted_output.predict)
        rmse_threshold = np.std(test_data.target) ** 2

        assert isinstance(pipeline, Pipeline)
        assert predicted_output.predict.shape[0] == n_samples * 0.2
        assert metric < rmse_threshold
