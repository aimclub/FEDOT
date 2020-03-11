import numpy as np
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData
from core.models.evaluation import SkLearnEvaluationStrategy
from core.models.model import Model
from core.repository.dataset_types import NumericalDataTypesEnum
from core.repository.model_types_repository import ModelTypesIdsEnum


@pytest.fixture()
def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)))

    return data


def model_by_type(model_type: ModelTypesIdsEnum):
    return Model(model_type=model_type,
                 input_type=NumericalDataTypesEnum.table,
                 output_type=NumericalDataTypesEnum.vector,
                 eval_strategy=SkLearnEvaluationStrategy())


@pytest.mark.skip(reason='The test should be refactored after EvoStrategy fix')
def test_log_regression_fit_correct(classification_dataset):
    data = classification_dataset
    log_reg = model_by_type(model_type=ModelTypesIdsEnum.logit)

    predicted = log_reg.evaluate(data=data)
    train_to = int(len(data.target) * 0.8)
    roc_on_train = roc_auc(y_true=data.target[:train_to, 0],
                           y_score=predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.skip(reason='The test should be refactored after EvoStrategy fix')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_random_forest_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    random_forest = model_by_type(model_type=ModelTypesIdsEnum.rf)

    random_forest.fit(data=data)
    predicted = random_forest.predict(data=data)

    train_to = int(len(predicted) * 0.8)

    roc_on_train = roc_auc(y_true=data.target[:train_to],
                           y_score=predicted[:train_to])
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.skip(reason='The test should be refactored after EvoStrategy fix')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_decision_tree_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    decision_tree = model_by_type(model_type=ModelTypesIdsEnum.dt)

    decision_tree.fit(data=data)
    predicted = decision_tree.predict(data=data)

    train_to = int(len(predicted) * 0.8)

    roc_on_train = roc_auc(y_true=data.target[:train_to],
                           y_score=predicted[:train_to])
    roc_threshold = 0.95

    assert roc_on_train >= roc_threshold


@pytest.mark.skip(reason='The test should be refactored after EvoStrategy fix')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_lda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    decision_tree = model_by_type(model_type=ModelTypesIdsEnum.lda)

    decision_tree.fit(data=data)
    predicted = decision_tree.predict(data=data)

    train_to = int(len(predicted) * 0.8)

    roc_on_train = roc_auc(y_true=data.target[:train_to],
                           y_score=predicted[:train_to])
    roc_threshold = 0.95

    assert roc_on_train >= roc_threshold


@pytest.mark.skip(reason='The test should be refactored after EvoStrategy fix')
@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_qda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    decision_tree = model_by_type(model_type=ModelTypesIdsEnum.qda)

    decision_tree.fit(data=data)
    predicted = decision_tree.predict(data=data)

    train_to = int(len(predicted) * 0.8)

    roc_on_train = roc_auc(y_true=data.target[:train_to],
                           y_score=predicted[:train_to])
    roc_threshold = 0.95

    assert roc_on_train >= roc_threshold
