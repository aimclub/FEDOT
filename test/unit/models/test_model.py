from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score as roc_auc
from sklearn.preprocessing import MinMaxScaler

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    PCAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.discriminant_analysis import \
    LDAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.naive import \
    NaiveAverageForecastImplementation, RepeatLastValueImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.operations.model import Model
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.operations.operation_parameters import get_default_params, OperationParameters
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from test.unit.common_tests import is_predict_ignores_target
from test.unit.data_operations.test_time_series_operations import synthetic_univariate_ts
from test.unit.tasks.test_forecasting import get_ts_data, get_ts_data_with_dt_idx
from test.unit.tasks.test_regression import get_synthetic_regression_data


def check_predict_correct(model, fitted_operation, test_data):
    return is_predict_ignores_target(
        predict_func=model.predict,
        predict_args={'fitted_operation': fitted_operation},
        data_arg_name='data',
        input_data=test_data,
    )


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


def get_lda_incorrect_data():
    """
    Problem arise when features contain only one column which "ideally" mapping with target
    """
    features = np.array([[1.0], [0.0], [1.0], [1.0], [1.0], [0.0]])
    target = np.array([[1], [0], [1], [1], [1], [0]])

    task = Task(TaskTypesEnum.classification)
    input_data = InputData(idx=[0, 1, 2, 3, 4, 5],
                           features=features,
                           target=target, task=task,
                           data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData(was_preprocessed=False))
    return input_data


def get_pca_incorrect_data():
    """ Generate wide table with number of features twice more than number of objects """
    features = np.array([[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                         [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                         [1.0, 5.0, 4.5, 0.5, 1.0, 1.5]])
    target = np.array([[1], [2], [3]])

    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=[0, 1, 2],
                           features=features,
                           target=target, task=task,
                           data_type=DataTypesEnum.table)
    return input_data


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


def classification_dataset_with_redundant_features(
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
def test_classification_models_fit_predict_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    roc_threshold = 0.95
    logger = default_log(prefix='default_test_logger')

    with OperationTypesRepository() as repo:
        model_names = repo.suitable_operation(task_type=TaskTypesEnum.classification,
                                              data_type=data.data_type,
                                              tags=['ml'])

    for model_name in model_names:
        logger.info(f"Test classification model: {model_name}.")
        model = Model(operation_type=model_name)
        fitted_operation, train_predicted = model.fit(params=None, data=train_data)
        test_pred = model.predict(fitted_operation=fitted_operation, data=test_data)
        roc_on_test = get_roc_auc(valid_data=test_data,
                                  predicted_data=test_pred)
        if model_name not in ['bernb', 'multinb']:
            assert roc_on_test >= roc_threshold
        else:
            assert roc_on_test >= 0.45

        assert check_predict_correct(model, fitted_operation, test_data)


def test_regression_models_fit_predict_correct():
    data = get_synthetic_regression_data(n_samples=100, random_state=42)
    train_data, test_data = train_test_data_setup(data)
    logger = default_log(prefix='default_test_logger')

    with OperationTypesRepository() as repo:
        model_names = repo.suitable_operation(task_type=TaskTypesEnum.regression,
                                              tags=['ml'])

    for model_name in model_names:
        logger.info(f"Test regression model: {model_name}.")
        model = Model(operation_type=model_name)

        fitted_operation, train_predicted = model.fit(params=OperationParameters(), data=train_data)
        test_pred = model.predict(fitted_operation=fitted_operation, data=test_data)
        rmse_value_test = mean_squared_error(y_true=test_data.target, y_pred=test_pred.predict)

        rmse_threshold = np.std(test_data.target) ** 2
        assert rmse_value_test < rmse_threshold
        assert check_predict_correct(model, fitted_operation, test_data)


def test_ts_models_fit_predict_correct():
    train_data, test_data = get_ts_data(forecast_length=5)
    logger = default_log(prefix='default_test_logger')

    with OperationTypesRepository() as repo:
        model_names = repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
                                              tags=['time_series'])

    for model_name in model_names:
        logger.info(f"Test time series model: {model_name}.")
        model = Model(operation_type=model_name)

        default_params = get_default_params(model_name)
        if not default_params:
            default_params = None

        fitted_operation, train_predicted = model.fit(params=OperationParameters(parameters=default_params), data=deepcopy(train_data))
        test_pred = model.predict(fitted_operation=fitted_operation, data=test_data)
        mae_value_test = mean_absolute_error(y_true=test_data.target, y_pred=test_pred.predict[0])

        mae_threshold = np.var(test_data.target) * 2
        assert mae_value_test < mae_threshold
        assert check_predict_correct(model, fitted_operation, test_data)


def test_ts_models_dt_idx_fit_correct():
    """Test to check if all time series models fit correct with datetime indexes"""
    train_data, test_data = get_ts_data_with_dt_idx(forecast_length=5)
    logger = default_log(prefix='default_test_logger')

    with OperationTypesRepository() as repo:
        model_names = repo.suitable_operation(task_type=TaskTypesEnum.ts_forecasting,
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

    first_node = PrimaryNode('normalization')
    second_node = SecondaryNode('kmeans', nodes_from=[first_node])
    pipeline = Pipeline(nodes=[first_node, second_node])
    train_predicted = pipeline.fit(train_data)

    assert all(np.unique(train_predicted.predict) == [0, 1])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    first_node = PrimaryNode('normalization')
    second_node = SecondaryNode('svc', nodes_from=[first_node])
    pipeline = Pipeline(nodes=[first_node, second_node])
    train_predicted = pipeline.fit(train_data)

    roc_on_train = get_roc_auc(valid_data=train_data,
                               predicted_data=train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


def test_pca_model_removes_redundant_features_correct():
    n_informative = 5
    data = classification_dataset_with_redundant_features(n_samples=100, n_features=10,
                                                          n_informative=n_informative)
    train_data, test_data = train_test_data_setup(data=data)

    first_node = PrimaryNode('normalization')
    second_node = SecondaryNode('pca', nodes_from=[first_node])
    pipeline = Pipeline(nodes=[first_node, second_node])
    train_predicted = pipeline.fit(train_data)
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
    glm_impl = GLMImplementation(OperationParameters(parameters={"family": "gaussian", "link": "identity"}))
    glm_impl.fit(input_data)
    predicted = glm_impl.predict_for_fit(input_data)
    pred_values = predicted.predict
    for i in range(9):
        assert pred_values[i, 0] - i < 0.5
        assert predicted.idx[i] - pred_values[i, 0] < 0.5


def test_lda_model_fit_with_incorrect_data():
    """
    Data is generated that for some versions of python (3.7) does not allow to
    train the LDA model with default parameters correctly.
    """
    lda_data = get_lda_incorrect_data()
    lda_model = LDAImplementation()
    lda_model.fit(lda_data)
    params = lda_model.get_params()

    assert 'solver' in params.changed_parameters.keys()


def test_pca_model_fit_with_wide_table():
    """
    The default value of n_components is 'mle' causes an error if the number of columns
    is greater than the number of rows in the dataset. If this happens, the number
    of n_components will be defined as 0.5
    """
    pca_data = get_pca_incorrect_data()
    pca_model = PCAImplementation()
    pca_model.fit(pca_data)

    params = pca_model.get_params()
    assert 'n_components' in params.changed_parameters.keys()


def test_ts_naive_average_forecast_correctly():
    """ Check if forecasted time series has correct indices """
    train_input, predict_input, _ = synthetic_univariate_ts()

    model = NaiveAverageForecastImplementation(OperationParameters(parameters={'part_for_averaging': 1.0}))
    fit_forecast = model.predict_for_fit(train_input)
    predict_forecast = model.predict(predict_input)

    # Check correctness during pipeline fit stage
    assert (10, 4) == fit_forecast.target.shape
    assert np.array_equal(fit_forecast.idx, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert np.isclose(fit_forecast.predict[0, 0], 0)

    # Pipeline predict stage
    assert np.array_equal(predict_forecast.predict, np.array([[65, 65, 65, 65]]))


def test_locf_forecast_correctly():
    """ Testing naive LOCF model """
    train_input, predict_input, _ = synthetic_univariate_ts()
    model = RepeatLastValueImplementation(OperationParameters(parameters={'part_for_repeat': 0.2}))

    model.fit(train_input)
    fit_forecast = model.predict_for_fit(train_input)
    predict_forecast = model.predict(predict_input)

    assert (8, 4) == fit_forecast.target.shape
    assert np.array_equal(fit_forecast.idx, np.array([3, 4, 5, 6, 7, 8, 9, 10]))
    # Repeated pattern (3 elements to repeat and 4 forecast horizon)
    assert np.array_equal(predict_forecast.predict, np.array([[110, 120, 130, 110]]))
