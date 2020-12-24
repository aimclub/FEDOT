from fedot.core.data.data import InputData
from fedot.core.models.evaluation.vectorize import VectorizeStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.feature_extraction.text import TfidfVectorizer


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

    vectorizer = VectorizeStrategy(model_type='tfidf', params=None)

    vectorizer_fitted = vectorizer.fit(train_data)

    predicted = vectorizer.predict(trained_model=vectorizer_fitted,
                                   predict_data=test_data)

    assert isinstance(vectorizer_fitted, TfidfVectorizer)
    assert len(predicted[0]) == 7
