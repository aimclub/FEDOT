from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score as roc_auc
from sklearn.preprocessing import MinMaxScaler

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PrimaryNode, get_default_params
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import DEFAULT_PARAMS_STUB
from test.unit.tasks.test_forecasting import get_ts_data, get_ts_data_with_dt_idx
from test.unit.tasks.test_regression import get_synthetic_regression_data


def get_roc_auc(valid_data: InputData, predicted_data: OutputData) -> float:
    n_classes = valid_data.num_classes
    if n_classes > 2:
        additional_params = {'multi_class': 'ovo', 'average': 'macro'}
    else:
        additional_params = {}

    try:
        roc_on_train = round(roc_auc(valid_data.target,
                                     predicted_data.predict,
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
    data = InputData(features=MinMaxScaler().fit_transform(x), target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def classification_dataset_with_redunant_features(
        n_samples=1000, n_features=100, n_informative=5) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative)

    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


def generate_simple_series():
    y = np.arange(11) + np.random.normal(loc=0, scale=0.1, size=11)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=2))
    i_d = InputData(idx=np.arange(11),
                    features=y,
                    target=y,
                    task=task,
                    data_type=DataTypesEnum.ts
                    )
    return i_d


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_classification_models_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    roc_threshold = 0.95
    logger = default_log('default_test_logger')

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.classification,
                                                 data_type=data.data_type,
                                                 tags=['ml'])

    for model_name in model_names:
        logger.info(f"Test classification model: {model_name}.")
        model = Model(operation_type=model_name)
        _, train_predicted = model.fit(params=None, data=train_data)
        test_pred = model.predict(fitted_operation=_, data=test_data, is_fit_pipeline_stage=False)
        roc_on_test = get_roc_auc(valid_data=test_data,
                                  predicted_data=test_pred)
        if model_name not in ['bernb', 'multinb']:
            assert roc_on_test >= roc_threshold
        else:
            assert roc_on_test >= 0.45


def test_regression_models_fit_correct():
    data = get_synthetic_regression_data(n_samples=1000, random_state=42)
    train_data, test_data = train_test_data_setup(data)
    logger = default_log('default_test_logger')

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.regression,
                                                 tags=['ml'])

    for model_name in model_names:
        logger.info(f"Test regression model: {model_name}.")
        model = Model(operation_type=model_name)

        _, train_predicted = model.fit(params=None, data=train_data)
        test_pred = model.predict(fitted_operation=_, data=test_data, is_fit_pipeline_stage=False)
        rmse_value_test = mean_squared_error(y_true=test_data.target, y_pred=test_pred.predict)

        rmse_threshold = np.std(test_data.target) ** 2
        assert rmse_value_test < rmse_threshold


def test_ts_models_fit_correct():
    train_data, test_data = get_ts_data(forecast_length=5)
    logger = default_log('default_test_logger')

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
                                                 tags=['time_series'])

    for model_name in model_names:
        logger.info(f"Test time series model: {model_name}.")
        model = Model(operation_type=model_name)

        default_params = get_default_params(model_name)
        if not default_params:
            default_params = None

        _, train_predicted = model.fit(params=default_params, data=deepcopy(train_data))
        test_pred = model.predict(fitted_operation=_, data=test_data, is_fit_pipeline_stage=False)
        mae_value_test = mean_absolute_error(y_true=test_data.target, y_pred=test_pred.predict[0])

        mae_threshold = np.var(test_data.target) * 2
        assert mae_value_test < mae_threshold


def test_ts_models_dt_idx_fit_correct():
    """Test to check if all time series models fit correct with datetime indexes"""
    train_data, test_data = get_ts_data_with_dt_idx(forecast_length=5)
    logger = default_log('default_test_logger')

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
                                                 tags=['time_series'])

    for model_name in model_names:
        logger.info(f"Test time series model: {model_name}.")
        node = PrimaryNode(model_name)
        pipeline = Pipeline(node)

        pipeline.fit(deepcopy(train_data))
        predicted = pipeline.predict(test_data)
        assert np.all(predicted.idx == test_data.idx)


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_log_clustering_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling pipeline. Fit predict it
    scaling_pipeline = Pipeline(PrimaryNode('normalization'))
    scaling_pipeline.fit(train_data)
    scaled_data = scaling_pipeline.predict(train_data)

    kmeans = Model(operation_type='kmeans')
    _, train_predicted = kmeans.fit(params=None, data=scaled_data)

    assert all(np.unique(train_predicted.predict) == [0, 1])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling pipeline. Fit predict it
    scaling_pipeline = Pipeline(PrimaryNode('normalization'))
    scaling_pipeline.fit(train_data)
    scaled_data = scaling_pipeline.predict(train_data)

    svc = Model(operation_type='svc')
    _, train_predicted = svc.fit(params=None, data=scaled_data)

    roc_on_train = get_roc_auc(valid_data=train_data,
                               predicted_data=train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


def test_pca_model_removes_redunant_features_correct():
    n_informative = 5
    data = classification_dataset_with_redunant_features(n_samples=1000, n_features=100,
                                                         n_informative=n_informative)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling pipeline. Fit predict it
    scaling_pipeline = Pipeline(PrimaryNode('normalization'))
    scaling_pipeline.fit(train_data)
    scaled_data = scaling_pipeline.predict(train_data)

    pca = DataOperation(operation_type='pca')
    _, train_predicted = pca.fit(params=DEFAULT_PARAMS_STUB, data=scaled_data)
    transformed_features = train_predicted.predict

    assert transformed_features.shape[1] < data.features.shape[1]


def test_glm_indexes_correct():
    """
    Test  checks correct indexing after performing non-lagged operation.
    For example we generate time series [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    and use GLM as estimator

    output should looks like
    [
    [0+N(0, 0.1), 1+N(0, 0.1)],
    [1+N(0, 0.1), 2+N(0, 0.1)],
    [2+N(0, 0.1), 3+N(0, 0.1)],
    [3+N(0, 0.1), 4+N(0, 0.1)],
    [4+N(0, 0.1), 5+N(0, 0.1)],
    [5+N(0, 0.1), 6+N(0, 0.1)],
    [6+N(0, 0.1), 7+N(0, 0.1)],
    [7+N(0, 0.1), 8+N(0, 0.1)],
    [8+N(0, 0.1 9+N(0, 0.1)]
    ]

    and indexes look like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    input_data = generate_simple_series()
    glm_impl = GLMImplementation(family="gaussian", link="identity")
    glm_impl.fit(input_data)
    predicted = glm_impl.predict(input_data, is_fit_pipeline_stage=True)
    pred_values = predicted.predict
    for i in range(9):
        assert pred_values[i, 0] - i < 0.5
        assert predicted.idx[i] - pred_values[i, 0] < 0.5
