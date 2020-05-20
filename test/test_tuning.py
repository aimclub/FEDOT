import os

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData, train_test_data_setup
from core.models.model import Model
from core.models.preprocessing import Scaling
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum
from test.test_autoregression import get_synthetic_ts_data


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/scoring_train_cat.csv'
    return InputData.from_csv(os.path.join(test_file_path, file))


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_knn_classification_tune_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    data.features = Scaling().fit(data.features).apply(data.features)
    data.task_type = MachineLearningTasksEnum.classification
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
