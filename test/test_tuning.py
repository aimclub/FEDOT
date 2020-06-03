import os

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as roc_auc

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData, train_test_data_setup
from core.models.model import Model
from core.models.preprocessing import Scaling
from core.repository.model_types_repository import ModelTypesIdsEnum
from test.test_autoregression import get_synthetic_ts_data


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/scoring_train_cat.csv'
    return InputData.from_csv(os.path.join(test_file_path, file))


@pytest.fixture()
def scoring_dataset():
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    return train_data, test_data


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_knn_classification_tune_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    knn = Model(model_type=ModelTypesIdsEnum.knn)
    model, _ = knn.fit(data=train_data)
    test_predicted = knn.predict(fitted_model=model, data=test_data)

    roc_on_test = roc_auc(y_true=test_data.target,
                          y_score=test_predicted)

    knn_for_tune = Model(model_type=ModelTypesIdsEnum.knn)
    model, _ = knn_for_tune.fine_tune(data=train_data, iterations=10)
    test_predicted_tuned = knn.predict(fitted_model=model, data=test_data)

    roc_on_test_tuned = roc_auc(y_true=test_data.target,
                                y_score=test_predicted_tuned)
    roc_threshold = 0.7
    assert roc_on_test_tuned > roc_on_test > roc_threshold


def test_arima_ar_tune_correct():
    data = get_synthetic_ts_data()
    train_data, test_data = train_test_data_setup(data=data)

    arima_for_tune = Model(model_type=ModelTypesIdsEnum.arima)
    model, _ = arima_for_tune.fine_tune(data=train_data, iterations=5)
    test_predicted_tuned = arima_for_tune.predict(fitted_model=model, data=test_data)

    rmse_on_test_tuned = mse(y_true=test_data.target,
                             y_pred=test_predicted_tuned, squared=False)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test_tuned < rmse_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_rf_class_tune_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    train_data, test_data = train_test_data_setup(data=data)

    rf = Model(model_type=ModelTypesIdsEnum.rf)

    model, _ = rf.fit(train_data)
    test_predicted = rf.predict(fitted_model=model, data=test_data)

    test_roc_auc = roc_auc(y_true=test_data.target,
                           y_score=test_predicted)

    model_tuned, _ = rf.fine_tune(data=train_data, iterations=12)
    test_predicted_tuned = rf.predict(fitted_model=model_tuned, data=test_data)

    test_roc_auc_tuned = roc_auc(y_true=test_data.target,
                                 y_score=test_predicted_tuned)
    roc_threshold = 0.7

    assert test_roc_auc_tuned != test_roc_auc
    assert test_roc_auc_tuned > roc_threshold


@pytest.mark.parametrize('data_fixture', ['scoring_dataset'])
def test_scoring_logreg_tune_correct(data_fixture, request):
    train_data, test_data = request.getfixturevalue(data_fixture)

    train_data.features = Scaling().fit(train_data.features).apply(train_data.features)
    test_data.features = Scaling().fit(test_data.features).apply(test_data.features)

    logreg = Model(model_type=ModelTypesIdsEnum.logit)

    model, _ = logreg.fit(train_data)
    test_predicted = logreg.predict(fitted_model=model, data=test_data)

    test_roc_auc = roc_auc(y_true=test_data.target,
                           y_score=test_predicted)

    logreg_for_tune = Model(model_type=ModelTypesIdsEnum.logit)

    model_tuned, _ = logreg_for_tune.fine_tune(train_data, iterations=50)
    test_predicted_tuned = logreg_for_tune.predict(fitted_model=model_tuned, data=test_data)

    test_roc_auc_tuned = roc_auc(y_true=test_data.target,
                                 y_score=test_predicted_tuned)

    roc_threshold = 0.6

    assert test_roc_auc_tuned >= test_roc_auc > roc_threshold
