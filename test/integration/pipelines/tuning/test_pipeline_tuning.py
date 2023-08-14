import os
from time import time

import pytest
from golem.core.tuning.hyperopt_tuner import get_node_parameters_for_hyperopt
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from hyperopt import hp
from hyperopt.pyll.stochastic import sample as hp_sample
from sklearn.metrics import mean_squared_error as mse, accuracy_score as acc

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum, ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root, NESTED_PARAMS_LABEL
from test.unit.tasks.test_forecasting import get_ts_data


@pytest.fixture(scope='package')
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(str(fedot_project_root()), 'test/data/simple_regression_train.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(str(fedot_project_root()), 'test/data/simple_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


@pytest.fixture()
def tiny_classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(str(fedot_project_root()), 'test/data/tiny_simple_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


@pytest.fixture()
def multi_classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(str(fedot_project_root()), 'test/data/multiclass_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


def get_simple_regr_pipeline(operation_type='rfr'):
    final = PipelineNode(operation_type=operation_type)
    pipeline = Pipeline(final)

    return pipeline


def get_complex_regr_pipeline():
    node_scaling = PipelineNode(operation_type='scaling')
    node_ridge = PipelineNode('ridge', nodes_from=[node_scaling])
    node_linear = PipelineNode('linear', nodes_from=[node_scaling])
    final = PipelineNode('rfr', nodes_from=[node_ridge, node_linear])
    pipeline = Pipeline(final)

    return pipeline


def get_regr_pipelines():
    simple_pipelines = [get_simple_regr_pipeline(operation_type) for operation_type in get_regr_operation_types()]

    return simple_pipelines + [get_complex_regr_pipeline()]


def get_simple_class_pipeline(operation_type='logit'):
    final = PipelineNode(operation_type=operation_type)
    pipeline = Pipeline(final)

    return pipeline


def get_complex_class_pipeline():
    first = PipelineNode(operation_type='knn')
    second = PipelineNode(operation_type='pca')
    final = PipelineNode(operation_type='logit',
                         nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


def get_pipeline_with_no_params_to_tune():
    first = PipelineNode(operation_type='scaling')
    final = PipelineNode(operation_type='bernb',
                         nodes_from=[first])

    pipeline = Pipeline(final)

    return pipeline


def get_class_pipelines():
    simple_pipelines = [get_simple_class_pipeline(operation_type) for operation_type in get_class_operation_types()]

    return simple_pipelines + [get_complex_class_pipeline()]


def get_regr_operation_types():
    return ['lgbmreg']


def get_class_operation_types():
    return ['dt']


def get_regr_losses():
    return [RegressionMetricsEnum.RMSE]


def get_class_losses():
    return [ClassificationMetricsEnum.ROCAUC, ClassificationMetricsEnum.accuracy]


def get_not_default_search_space():
    custom_search_space = {
        'logit': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-1, 5.0],
                'type': 'continuous'}
        },
        'ridge': {
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 5.0],
                'type': 'continuous'}
        },
        'lgbmreg': {
            'learning_rate': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [0.05, 0.1],
                'type': 'continuous'},
            'colsample_bytree': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.2, 0.8],
                'type': 'continuous'},
            'subsample': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.8],
                'type': 'continuous'}
        },
        'dt': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 5],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [10, 25],
                'type': 'discrete'}
        },
        'ar': {
            'lag_1': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [2, 100],
                'type': 'continuous'},
            'lag_2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [2, 500],
                'type': 'continuous'}
        },
        'pca': {
            'n_components': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.5],
                'type': 'continuous'}
        }
    }
    return PipelineSearchSpace(custom_search_space=custom_search_space)


def custom_maximized_metrics(real_data: InputData, pred_data: OutputData):
    mse_value = mse(real_data.target, pred_data.predict, squared=False)
    return -(mse_value + 2) * 0.5


def custom_minimized_metrics(real_data: InputData, pred_data: OutputData):
    acc_value = acc(real_data.target, pred_data.predict)
    return 100 - (acc_value + 2) * 0.5


def run_pipeline_tuner(train_data,
                       pipeline,
                       loss_function,
                       tuner=SimultaneousTuner,
                       search_space=PipelineSearchSpace(),
                       cv=None,
                       iterations=1,
                       early_stopping_rounds=None):
    # Pipeline tuning
    pipeline_tuner = TunerBuilder(train_data.task) \
        .with_tuner(tuner) \
        .with_metric(loss_function) \
        .with_cv_folds(cv) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_search_space(search_space) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(pipeline)
    return pipeline_tuner, tuned_pipeline


def run_node_tuner(train_data,
                   pipeline,
                   loss_function,
                   search_space=PipelineSearchSpace(),
                   cv=None,
                   node_index=0,
                   iterations=1,
                   early_stopping_rounds=None):
    # Pipeline tuning
    node_tuner = TunerBuilder(train_data.task) \
        .with_tuner(SequentialTuner) \
        .with_metric(loss_function) \
        .with_cv_folds(cv) \
        .with_iterations(iterations) \
        .with_search_space(search_space) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .build(train_data)
    tuned_pipeline = node_tuner.tune_node(pipeline, node_index)
    return node_tuner, tuned_pipeline


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    pipeline = get_complex_class_pipeline()

    custom_params = dict(C=10)

    pipeline.root_node.parameters = custom_params
    pipeline.fit(data)
    params = pipeline.root_node.fitted_operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_pipeline_tuner_correct(data_fixture, pipelines, loss_functions, request, tuner):
    """ Test all tuners for pipeline """
    data = request.getfixturevalue(data_fixture)
    cvs = [None, 2]

    for pipeline in pipelines:
        for loss_function in loss_functions:
            for cv in cvs:
                pipeline_tuner, tuned_pipeline = run_pipeline_tuner(tuner=tuner,
                                                                    train_data=data,
                                                                    pipeline=pipeline,
                                                                    loss_function=loss_function,
                                                                    cv=cv)
                assert pipeline_tuner.obtained_metric is not None
                assert not tuned_pipeline.is_fitted

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_pipeline_tuner_with_no_parameters_to_tune(classification_dataset, tuner):
    pipeline = get_pipeline_with_no_params_to_tune()
    pipeline_tuner, tuned_pipeline = run_pipeline_tuner(tuner=tuner,
                                                        train_data=classification_dataset,
                                                        pipeline=pipeline,
                                                        loss_function=ClassificationMetricsEnum.ROCAUC,
                                                        iterations=20)
    assert pipeline_tuner.obtained_metric is not None
    assert pipeline_tuner.obtained_metric == pipeline_tuner.init_metric
    assert not tuned_pipeline.is_fitted


@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_pipeline_tuner_with_initial_params(classification_dataset, tuner):
    """ Test all tuners for pipeline with initial parameters """
    # a model
    node = PipelineNode(content={'name': 'xgboost', 'params': {'max_depth': 3,
                                                               'learning_rate': 0.03,
                                                               'min_child_weight': 2}})
    pipeline = Pipeline(node)
    pipeline_tuner, tuned_pipeline = run_pipeline_tuner(tuner=tuner,
                                                        train_data=classification_dataset,
                                                        pipeline=pipeline,
                                                        loss_function=ClassificationMetricsEnum.ROCAUC,
                                                        iterations=20)
    assert pipeline_tuner.obtained_metric is not None
    assert not tuned_pipeline.is_fitted


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_pipeline_tuner_with_custom_search_space(data_fixture, pipelines, loss_functions, request, tuner):
    """ Test tuners with different search spaces """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    search_spaces = [PipelineSearchSpace(), get_not_default_search_space()]

    for search_space in search_spaces:
        pipeline_tuner, _ = run_pipeline_tuner(tuner=tuner,
                                               train_data=train_data,
                                               pipeline=pipelines[0],
                                               loss_function=loss_functions[0],
                                               search_space=search_space)
        assert pipeline_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_certain_node_tuning_correct(data_fixture, pipelines, loss_functions, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    cvs = [None, 2]

    for pipeline in pipelines:
        for loss_function in loss_functions:
            for cv in cvs:
                node_tuner, tuned_pipeline = run_node_tuner(train_data=data,
                                                            pipeline=pipeline,
                                                            loss_function=loss_function,
                                                            cv=cv)
                assert node_tuner.obtained_metric is not None
                assert not tuned_pipeline.is_fitted

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_certain_node_tuner_with_custom_search_space(data_fixture, pipelines, loss_functions, request):
    """ Test SequentialTuner for particular node with different search spaces """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    search_spaces = [PipelineSearchSpace(), get_not_default_search_space()]

    for search_space in search_spaces:
        node_tuner, _ = run_node_tuner(train_data=train_data,
                                       pipeline=pipelines[0],
                                       loss_function=loss_functions[0],
                                       search_space=search_space)
        assert node_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('n_steps', [100, 133, 217, 300])
@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_ts_pipeline_with_stats_model(n_steps, tuner):
    """ Tests tuners for time series forecasting task with AR model """
    train_data, test_data = get_ts_data(n_steps=n_steps, forecast_length=5)

    ar_pipeline = Pipeline(PipelineNode('ar'))

    for search_space in [PipelineSearchSpace(), get_not_default_search_space()]:
        # Tune AR model
        tuner_ar = TunerBuilder(train_data.task) \
            .with_tuner(tuner) \
            .with_metric(RegressionMetricsEnum.MSE) \
            .with_iterations(3) \
            .with_search_space(search_space).build(train_data)
        _ = tuner_ar.tune(ar_pipeline)

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['tiny_classification_dataset'])
def test_early_stop_in_tuning(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    start_pipeline_tuner = time()
    _ = run_pipeline_tuner(tuner=SimultaneousTuner,
                           train_data=train_data,
                           pipeline=get_class_pipelines()[0],
                           loss_function=ClassificationMetricsEnum.ROCAUC,
                           iterations=1000,
                           early_stopping_rounds=1)
    assert time() - start_pipeline_tuner < 1

    start_sequential_tuner = time()
    _ = run_pipeline_tuner(tuner=SequentialTuner,
                           train_data=train_data,
                           pipeline=get_class_pipelines()[0],
                           loss_function=ClassificationMetricsEnum.ROCAUC,
                           iterations=1000,
                           early_stopping_rounds=1)
    assert time() - start_sequential_tuner < 1

    start_node_tuner = time()
    _ = run_node_tuner(train_data=train_data,
                       pipeline=get_class_pipelines()[0],
                       loss_function=ClassificationMetricsEnum.ROCAUC,
                       iterations=1000,
                       early_stopping_rounds=1)
    assert time() - start_node_tuner < 1


def test_search_space_correctness_after_customization():
    default_search_space = PipelineSearchSpace()

    custom_search_space = {'gbr': {'max_depth': {
        'hyperopt-dist': hp.choice,
        'sampling-scope': [[3, 7, 31, 127, 8191, 131071]],
        'type': 'categorical'}}}
    custom_search_space_without_replace = PipelineSearchSpace(custom_search_space=custom_search_space,
                                                              replace_default_search_space=False)
    custom_search_space_with_replace = PipelineSearchSpace(custom_search_space=custom_search_space,
                                                           replace_default_search_space=True)

    default_params = get_node_parameters_for_hyperopt(default_search_space,
                                                      node_id=0,
                                                      operation_name='gbr')
    custom_without_replace_params = get_node_parameters_for_hyperopt(custom_search_space_without_replace,
                                                                     node_id=0,
                                                                     operation_name='gbr')
    custom_with_replace_params = get_node_parameters_for_hyperopt(custom_search_space_with_replace,
                                                                  node_id=0,
                                                                  operation_name='gbr')

    assert default_params.keys() == custom_without_replace_params.keys()
    assert default_params.keys() != custom_with_replace_params.keys()
    assert default_params['0 || gbr | max_depth'] != custom_without_replace_params['0 || gbr | max_depth']
    assert default_params['0 || gbr | max_depth'] != custom_with_replace_params['0 || gbr | max_depth']


def test_search_space_get_operation_parameter_range():
    default_search_space = PipelineSearchSpace()
    gbr_operations = ['loss', 'learning_rate', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'subsample', 'max_features', 'alpha']

    custom_search_space = {'gbr': {'max_depth': {
        'hyperopt-dist': hp.choice,
        'sampling-scope': [[3, 7, 31, 127, 8191, 131071]],
        'type': 'categorical'}}}
    custom_search_space_without_replace = PipelineSearchSpace(custom_search_space=custom_search_space,
                                                              replace_default_search_space=False)
    custom_search_space_with_replace = PipelineSearchSpace(custom_search_space=custom_search_space,
                                                           replace_default_search_space=True)

    default_operations = default_search_space.get_parameters_for_operation('gbr')
    custom_without_replace_operations = custom_search_space_without_replace.get_parameters_for_operation('gbr')
    custom_with_replace_operations = custom_search_space_with_replace.get_parameters_for_operation('gbr')

    assert default_operations == gbr_operations
    assert custom_without_replace_operations == gbr_operations
    assert custom_with_replace_operations == ['max_depth']


def test_complex_search_space():
    space = PipelineSearchSpace()
    for i in range(20):
        operation_parameters = space.parameters_per_operation.get("glm")
        new_value = hp_sample(operation_parameters[NESTED_PARAMS_LABEL])
        for params in new_value['sampling-scope'][0]:
            assert params['link'] in GLMImplementation.family_distribution[params['family']]['available_links']


@pytest.mark.parametrize('tuner', [SimultaneousTuner, SequentialTuner, IOptTuner])
def test_complex_search_space_tuning_correct(tuner):
    """ Tests SimultaneousTuner for time series forecasting task with GLM model that has a complex glm search space"""
    train_data, test_data = get_ts_data(n_steps=700, forecast_length=20)

    glm_pipeline = Pipeline(PipelineNode('glm'))
    initial_parameters = glm_pipeline.nodes[0].parameters
    tuner = TunerBuilder(train_data.task) \
        .with_tuner(tuner) \
        .with_metric(RegressionMetricsEnum.MSE) \
        .with_iterations(100) \
        .build(train_data)
    tuned_glm_pipeline = tuner.tune(glm_pipeline)
    found_parameters = tuned_glm_pipeline.nodes[0].parameters
    if tuner.init_metric == tuner.obtained_metric:
        # TODO: (YamLyubov) Remove the check when IOptTuner will be able to tune categorical parameters.
        assert initial_parameters == found_parameters
    else:
        assert initial_parameters != found_parameters
