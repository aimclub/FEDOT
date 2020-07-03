import numpy as np
import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData, train_test_data_setup
from core.models.model import Model
from core.models.preprocessing import Scaling
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.tasks import Task, TaskTypesEnum


def get_roc_auc(train_data: InputData, train_predicted: list) -> float:
    n_classes = train_data.num_classes
    if n_classes > 2:
        additional_params = {'multi_class': 'ovo', 'average': 'macro'}
    else:
        additional_params = {}

    try:
        roc_on_train = round(roc_auc(y_score=train_predicted,
                                     y_true=train_data.target,
                                     **additional_params), 3)
    except Exception as ex:
        print(ex)
        roc_on_train = 0.5

    return roc_on_train


@pytest.fixture()
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


def test_log_regression_fit_correct(classification_dataset):
    data = classification_dataset
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    log_reg = Model(model_type=ModelTypesIdsEnum.logit)

    _, train_predicted = log_reg.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_random_forest_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    random_forest = Model(model_type=ModelTypesIdsEnum.rf)

    _, train_predicted = random_forest.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_decision_tree_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    decision_tree = Model(model_type=ModelTypesIdsEnum.dt)

    decision_tree.fit(data=train_data)
    _, train_predicted = decision_tree.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_lda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    lda = Model(model_type=ModelTypesIdsEnum.lda)

    _, train_predicted = lda.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_qda_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    qda = Model(model_type=ModelTypesIdsEnum.qda)

    _, train_predicted = qda.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_log_clustering_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    kmeans = Model(model_type=ModelTypesIdsEnum.kmeans)

    _, train_predicted = kmeans.fit(data=train_data)

    assert all(np.unique(train_predicted) == [0, 1])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    svc = Model(model_type=ModelTypesIdsEnum.svc)

    _, train_predicted = svc.fit(data=train_data)

    roc_on_train = get_roc_auc(train_data, train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_sklearn_model_set_custom_params(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    custom_params = dict(n_neighbors=10,
                         weights='uniform',
                         p=1)

    model = Model(model_type=ModelTypesIdsEnum.knn)

    model.set_custom_params(custom_params)

    fitted_model, _ = model.fit(data=data)
    model_params = fitted_model.get_params()

    assert model_params.get('n_neighbors') == 10
