import pickle
from copy import deepcopy
from time import perf_counter
from typing import Tuple, Optional

import numpy as np
import pytest

from fedot.api.api_utils.presets import PresetsEnum
from fedot.core.repository.operation_tags_n_repo_enums import OtherTagsEnum
from fedot.core.repository.operation_types_repo_enum import OperationReposEnum
from golem.core.log import default_log
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score as roc_auc
from sklearn.preprocessing import MinMaxScaler

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    PCAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.discriminant_analysis import \
    LDAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.naive import \
    NaiveAverageForecastImplementation, RepeatLastValueImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.operations.model import Model
from fedot.core.operations.operation_parameters import get_default_params, OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationMetaInfo, OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
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


def get_data_for_testing(task_type, data_type, length=100, features_count=1,
                         value=0, random=True):
    allowed_data_type = {TaskTypesEnum.ts_forecasting: [DataTypesEnum.ts, DataTypesEnum.multi_ts],
                         TaskTypesEnum.classification: [DataTypesEnum.table],
                         TaskTypesEnum.regression: [DataTypesEnum.table]}
    if task_type not in allowed_data_type or data_type not in allowed_data_type[task_type]:
        return None

    if task_type is TaskTypesEnum.ts_forecasting:
        task = Task(task_type, TsForecastingParams(max(length // 10, 2)))
        if data_type is DataTypesEnum.ts:
            features = np.zeros(length) + value
        else:
            features = np.zeros((length, features_count)) + value
        if data_type is DataTypesEnum.table:
            target = np.zeros(length) + value
        else:
            target = features

    else:
        task = Task(task_type)
        data_type = DataTypesEnum.table
        features = np.zeros((length, features_count)) + value
        target = np.zeros(length) + value
        if task_type is TaskTypesEnum.classification:
            target[:int(len(target) // 2)] = 2 * value + 1

    if random:
        generator = np.random.RandomState(value)
        features += generator.rand(*features.shape)
        if task_type is TaskTypesEnum.ts_forecasting:
            target = features
        elif task_type is not TaskTypesEnum.classification:
            target += generator.rand(*target.shape)

    data = InputData(idx=np.arange(length),
                     features=features,
                     target=target,
                     data_type=data_type,
                     task=task)
    return data


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
                           supplementary_data=SupplementaryData())
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


def get_operation_perfomance(operation: OperationMetaInfo,
                             data_lengths: Tuple[float, ...],
                             times: int = 1) -> Optional[Tuple[float, ...]]:
    """
    Helper function to check perfomance of only the first valid operation pair (task_type, input_type).
    """
    def fit_time_for_operation(operation: OperationMetaInfo,
                               data: InputData):
        nodes_from = []
        if task_type is TaskTypesEnum.ts_forecasting:
            if 'non_lagged' not in operation.tags:
                nodes_from = [PipelineNode('lagged')]
        node = PipelineNode(operation.id, nodes_from=nodes_from)
        pipeline = Pipeline(node)
        start_time = perf_counter()
        pipeline.fit(data)
        return perf_counter() - start_time

    for task_type in operation.task_type:
        for data_type in operation.input_types:
            perfomance_values = []
            for length in data_lengths:
                data = get_data_for_testing(task_type, data_type,
                                            length=length, features_count=2,
                                            random=True)
                if data is not None:
                    min_evaluated_time = min(fit_time_for_operation(operation, data) for _ in range(times))
                    perfomance_values.append(min_evaluated_time)
            if perfomance_values:
                if len(perfomance_values) != len(data_lengths):
                    raise ValueError('not all measurements have been proceeded')
                return tuple(perfomance_values)
    raise Exception(f"Fit time for operation ``{operation.id}`` cannot be measured")


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


def classification_dataset_with_str_labels():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array(['a' if val <= threshold else 'b' for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=MinMaxScaler().fit_transform(x), target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def classification_dataset_with_redundant_features(
        n_samples=1000, n_features=100, n_informative=5) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative, random_state=42)

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
                                              tags=[OtherTagsEnum.ml])

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
                                              tags=[OtherTagsEnum.ml])

    for model_name in model_names:
        logger.info(f"Test regression model: {model_name}.")
        model = Model(operation_type=model_name)

        fitted_operation, train_predicted = model.fit(params=None, data=train_data)
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
                                              tags=[OtherTagsEnum.non_lagged])

    for model_name in model_names:
        logger.info(f"Test time series model: {model_name}.")
        model = Model(operation_type=model_name)

        default_params = get_default_params(model_name)
        if not default_params:
            default_params = None

        fitted_operation, train_predicted = model.fit(params=default_params,
                                                      data=deepcopy(train_data))
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
                                              tags=[OtherTagsEnum.non_lagged])

    for model_name in model_names:
        logger.info(f"Test time series model: {model_name}.")
        node = PipelineNode(model_name)
        pipeline = Pipeline(node)

        pipeline.fit(deepcopy(train_data))
        predicted = pipeline.predict(test_data)
        assert np.all(predicted.idx == test_data.idx)


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_log_clustering_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    first_node = PipelineNode('normalization')
    second_node = PipelineNode('kmeans', nodes_from=[first_node])
    pipeline = Pipeline(nodes=[first_node, second_node])
    train_predicted = pipeline.fit(train_data)

    assert all(np.unique(train_predicted.predict) == [0, 1])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    first_node = PipelineNode('normalization')
    second_node = PipelineNode('svc', nodes_from=[first_node])
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

    first_node = PipelineNode('normalization')
    second_node = PipelineNode('pca', nodes_from=[first_node])
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
    glm_impl = GLMImplementation(OperationParameters(**{"family": "gaussian", "link": "identity"}))
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

    model = NaiveAverageForecastImplementation(OperationParameters(part_for_averaging=1.0))
    fit_forecast = model.predict_for_fit(train_input)
    predict_forecast = model.predict(predict_input)

    # Check correctness during pipeline fit stage
    assert (11, 4) == fit_forecast.target.shape
    assert np.array_equal(fit_forecast.idx, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert np.isclose(fit_forecast.predict[0, 0], 0)

    # Pipeline predict stage
    assert np.array_equal(predict_forecast.predict, np.array([[65, 65, 65, 65]]))


def test_locf_forecast_correctly():
    """ Testing naive LOCF model """
    train_input, predict_input, _ = synthetic_univariate_ts()
    model = RepeatLastValueImplementation(OperationParameters(part_for_repeat=0.2))

    model.fit(train_input)
    fit_forecast = model.predict_for_fit(train_input)
    predict_forecast = model.predict(predict_input)

    assert (8, 4) == fit_forecast.target.shape
    assert np.array_equal(fit_forecast.idx, np.array([3, 4, 5, 6, 7, 8, 9, 10]))
    # Repeated pattern (3 elements to repeat and 4 forecast horizon)
    assert np.array_equal(predict_forecast.predict, np.array([[110, 120, 130, 110]]))


def test_models_does_not_fall_on_constant_data():
    """ Run models on constant data """
    # models that raise exception
    to_skip = ['custom', 'arima', 'catboost', 'catboostreg', 'cgru',
               'lda', 'fast_ica', 'decompose', 'class_decompose']
    to_skip += ['sgd', 'elasticnet', 'minibatchsgd', 'mbsgdcregr', 'cd']  # TODO enable after gpu preset correct tuning

    for operation in OperationTypesRepository(OperationReposEnum.ALL).repo:
        if operation.id in to_skip:
            continue
        for task_type in operation.task_type:
            for data_type in operation.input_types:
                data = get_data_for_testing(task_type, data_type,
                                            length=100, features_count=2,
                                            random=False)
                if data is not None:

                    nodes_from = []
                    if task_type is TaskTypesEnum.ts_forecasting:
                        if OtherTagsEnum.non_lagged not in operation.tags:
                            nodes_from = [PipelineNode('lagged')]
                    node = PipelineNode(operation.id, nodes_from=nodes_from)
                    pipeline = Pipeline(node)
                    pipeline.fit(data)
                    assert pipeline.predict(data) is not None


def test_operations_are_serializable():
    to_skip = ['custom', 'decompose', 'class_decompose']
    to_skip += ['sgd', 'elasticnet', 'minibatchsgd', 'mbsgdcregr', 'cd']  # TODO enable after gpu preset correct tuning

    for operation in OperationTypesRepository(OperationReposEnum.ALL).repo:
        if operation.id in to_skip:
            continue
        for task_type in operation.task_type:
            for data_type in operation.input_types:
                data = get_data_for_testing(task_type, data_type,
                                            length=100, features_count=2,
                                            random=True)
                if data is not None:
                    try:
                        nodes_from = []
                        if task_type is TaskTypesEnum.ts_forecasting:
                            if OtherTagsEnum.non_lagged not in operation.tags:
                                nodes_from = [PipelineNode('lagged')]
                        node = PipelineNode(operation.id, nodes_from=nodes_from)
                        pipeline = Pipeline(node)
                        pipeline.fit(data)
                        serialized = pickle.dumps(pipeline, pickle.HIGHEST_PROTOCOL)
                        assert isinstance(serialized, bytes)
                    except NotImplementedError:
                        pass


def test_operations_are_fast():
    """
    Test ensures that all operations with fast_train preset meet sustainability expectation.
    Test defines operation complexity as polynomial function of data size.
    If complexity function grows fast, then operation should not have fast_train tag.
    """

    data_lengths = tuple(map(int, np.logspace(2.2, 4, 6)))
    reference_operations = ['rf', 'rfr']
    to_skip = ['custom', 'decompose', 'class_decompose', 'kmeans',
               'resample', 'one_hot_encoding'] + reference_operations
    to_skip += ['sgd', 'elasticnet', 'minibatchsgd', 'mbsgdcregr', 'cd']  # TODO enable after gpu preset correct tuning
    reference_time = (float('inf'), ) * len(data_lengths)
    # tries for time measuring
    attempt = 2

    for operation in OperationTypesRepository(OperationReposEnum.ALL).repo:
        if operation.id in reference_operations:
            perfomance_values = get_operation_perfomance(operation, data_lengths, attempt)
            reference_time = tuple(map(min, zip(perfomance_values, reference_time)))

    for operation in OperationTypesRepository(OperationReposEnum.ALL).repo:
        if (operation.id not in to_skip and operation.presets and PresetsEnum.FAST_TRAIN in operation.presets):
            for _ in range(attempt):
                perfomance_values = get_operation_perfomance(operation, data_lengths)
                # if attempt is successful then stop
                if all(x >= y for x, y in zip(reference_time, perfomance_values)):
                    break
            else:
                raise Exception(f"Operation {operation.id} cannot have ``fast-train`` tag")


def test_all_operations_are_documented():
    # All operations and presets should be listed in `docs/source/introduction/fedot_features/automation_features.rst`
    to_skip = {'custom', 'data_source_img', 'data_source_text', 'data_source_table', 'data_source_ts', 'exog_ts'}
    path_to_docs = fedot_project_root() / 'docs/source/introduction/fedot_features/automation_features.rst'
    docs_lines = None

    with open(path_to_docs, 'r') as docs_:
        docs_lines = docs_.readlines()
    if docs_lines:
        # TODO change DEFAULT to ALL after gpu repo fixing
        for operation in OperationTypesRepository(OperationReposEnum.DEFAULT).repo:
            if operation.id not in to_skip:
                for line in docs_lines:
                    if operation.id in line and all(preset in line for preset in operation.presets):
                        break
                else:
                    raise Exception(f"Operation {operation.id} with presets {operation.presets} \
                                    are not documented in {path_to_docs}")
