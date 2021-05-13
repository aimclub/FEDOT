import numpy as np
from sklearn.model_selection import KFold

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.cross_validation import cross_validation
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def sample_chain():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='scaling')
    final = SecondaryNode(operation_type='logit', nodes_from=[first, second])
    chain = Chain(final)

    return chain


def test_sklearn_kfold_correct():
    features = list('qwerasdfzxcvghkj')
    target = list('qwerasdfzxcvghkj')
    kf = KFold(n_splits=3)
    for train_idxs, test_idxs in kf.split(X=features, y=target):
        # train_values = [array[idx] for idx in train_idxs]
        # train_values = [array[idx] for idx in test_idxs]

        print(train_idxs)


def test_kfold_cv_metric_correct():
    source = classification_dataset()
    chain = sample_chain()

    actual_value = cross_validation(chain, source, cv=10, metrics=['roc_auc', 'precision'])

    assert actual_value > 0
