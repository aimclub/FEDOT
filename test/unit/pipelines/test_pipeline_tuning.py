import os
from random import seed

import numpy as np
import pytest
from hyperopt import hp, tpe, rand
from hyperopt.pyll.stochastic import sample as hp_sample
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc, accuracy_score as acc

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.tuner_interface import _greater_is_better
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.tasks.test_forecasting import get_ts_data

seed(1)
np.random.seed(1)


@pytest.fixture()
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_regression.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'advanced_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


def get_simple_regr_pipeline():
    final = PrimaryNode(operation_type='rfr')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_regr_pipeline():
    node_scaling = PrimaryNode(operation_type='scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_scaling])
    final = SecondaryNode('rfr', nodes_from=[node_ridge, node_linear])
    pipeline = Pipeline(final)

    return pipeline


def get_simple_class_pipeline():
    final = PrimaryNode(operation_type='logit')
    pipeline = Pipeline(final)

    return pipeline


def get_complex_class_pipeline():
    first = PrimaryNode(operation_type='xgboost')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

    pipeline = Pipeline(final)

    return pipeline


def get_not_default_search_space():
    custom_search_space = {
        'logit': {
            'C': (hp.uniform, [0.01, 5.0])
        },
        'ridge': {
            'alpha': (hp.uniform, [0.01, 5.0])
        },
        'xgbreg': {
            'n_estimators': (hp.choice, [[100]]),
            'max_depth': (hp.choice, [range(1, 7)]),
            'learning_rate': (hp.choice, [[1e-3, 1e-2, 1e-1]]),
            'subsample': (hp.choice, [np.arange(0.15, 1.01, 0.05)])
        },
        'xgboost': {
            'max_depth': (hp.choice, [range(1, 5)]),
            'subsample': (hp.uniform, [0.1, 0.9]),
            'min_child_weight': (hp.choice, [range(1, 15)])
        },
        'ar': {
            'lag_1': (hp.uniform, [2, 100]),
            'lag_2': (hp.uniform, [2, 500])
        },
        'pca': {
            'n_components': (hp.uniform, [0.2, 0.8])
        }
    }
    return SearchSpace(custom_search_space=custom_search_space)


def custom_maximized_metrics(y_true, y_pred):
    mse_value = mse(y_true, y_pred, squared=False)
    return -(mse_value + 2) * 0.5


def custom_minimized_metrics(y_true, y_pred):
    acc_value = acc(y_true, y_pred)
    return 100 - (acc_value + 2) * 0.5


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    pipeline = get_complex_class_pipeline()

    custom_params = dict(C=10)

    pipeline.root_node.custom_params = custom_params
    pipeline.fit(data)
    params = pipeline.root_node.fitted_operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_pipeline_tuner_regression_correct(data_fixture, request):
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1,
                                           search_space=search_space,
                                           algo=tpe.suggest)
            # Optimization will be performed on RMSE metric, so loss params are defined
            tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                          loss_function=mse,
                                                          loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_pipeline_tuner_classification_correct(data_fixture, request):
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1,
                                           search_space=search_space,
                                           algo=tpe.suggest)
            tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                          loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_sequential_tuner_regression_correct(data_fixture, request):
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            sequential_tuner = SequentialTuner(pipeline=pipeline,
                                               task=train_data.task,
                                               iterations=1,
                                               search_space=search_space,
                                               algo=tpe.suggest)
            # Optimization will be performed on RMSE metric, so loss params are defined
            tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                            loss_function=mse,
                                                            loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_sequential_tuner_classification_correct(data_fixture, request):
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            sequential_tuner = SequentialTuner(pipeline=pipeline,
                                               task=train_data.task,
                                               iterations=2,
                                               search_space=search_space,
                                               algo=tpe.suggest)
            tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                            loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_certain_node_tuning_regression_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            sequential_tuner = SequentialTuner(pipeline=pipeline,
                                               task=train_data.task,
                                               iterations=1,
                                               search_space=search_space,
                                               algo=tpe.suggest)
            tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                        node_index=0,
                                                        loss_function=mse)
    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_certain_node_tuning_classification_correct(data_fixture, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        for search_space in [SearchSpace(), get_not_default_search_space()]:
            # Pipeline tuning
            sequential_tuner = SequentialTuner(pipeline=pipeline,
                                               task=train_data.task,
                                               iterations=1,
                                               search_space=search_space,
                                               algo=tpe.suggest)
            tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                        node_index=0,
                                                        loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_ts_pipeline_with_stats_model():
    """ Tests PipelineTuner for time series forecasting task with AR model """
    train_data, test_data = get_ts_data(n_steps=200, forecast_length=5)

    ar_pipeline = Pipeline(PrimaryNode('ar'))

    for search_space in [SearchSpace(), get_not_default_search_space()]:
        # Tune AR model
        tuner_ar = PipelineTuner(pipeline=ar_pipeline, task=train_data.task, iterations=3,
                                 search_space=search_space, algo=rand.suggest)
        tuned_ar_pipeline = tuner_ar.tune_pipeline(input_data=train_data,
                                                   loss_function=mse)

    is_tuning_finished = True

    assert is_tuning_finished


def test_search_space_correctness_after_customization():
    default_search_space = SearchSpace()

    custom_search_space = {'gbr': {'max_depth': (hp.choice, [[3, 7, 31, 127, 8191, 131071]])}}
    custom_search_space_without_replace = SearchSpace(custom_search_space=custom_search_space,
                                                      replace_default_search_space=False)
    custom_search_space_with_replace = SearchSpace(custom_search_space=custom_search_space,
                                                   replace_default_search_space=True)

    default_params = default_search_space.get_node_params(node_id=0,
                                                          operation_name='gbr')
    custom_without_replace_params = custom_search_space_without_replace.get_node_params(node_id=0,
                                                                                        operation_name='gbr')
    custom_with_replace_params = custom_search_space_with_replace.get_node_params(node_id=0,
                                                                                  operation_name='gbr')

    assert default_params.keys() == custom_without_replace_params.keys()
    assert default_params.keys() != custom_with_replace_params.keys()
    assert default_params['0 || gbr | max_depth'] != custom_without_replace_params['0 || gbr | max_depth']
    assert default_params['0 || gbr | max_depth'] != custom_with_replace_params['0 || gbr | max_depth']


def test_search_space_get_operation_parameter_range():
    default_search_space = SearchSpace()
    gbr_operations = ['n_estimators', 'loss', 'learning_rate', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'subsample', 'max_features', 'alpha']

    custom_search_space = {'gbr': {'max_depth': (hp.choice, [[3, 7, 31, 127, 8191, 131071]])}}
    custom_search_space_without_replace = SearchSpace(custom_search_space=custom_search_space,
                                                      replace_default_search_space=False)
    custom_search_space_with_replace = SearchSpace(custom_search_space=custom_search_space,
                                                   replace_default_search_space=True)

    default_operations = default_search_space.get_operation_parameter_range('gbr')
    custom_without_replace_operations = custom_search_space_without_replace.get_operation_parameter_range('gbr')
    custom_with_replace_operations = custom_search_space_with_replace.get_operation_parameter_range('gbr')

    assert default_operations == gbr_operations
    assert custom_without_replace_operations == gbr_operations
    assert custom_with_replace_operations == ['max_depth']


def test_complex_search_space():
    space = SearchSpace()
    for i in range(20):
        operation_parameters = space.parameters_per_operation.get("glm")
        new_value = hp_sample(operation_parameters["nested_space"])
        for params in new_value[1][0]:
            assert params['link'] in GLMImplementation.family_distribution[params['family']]['available_links']


def test_complex_search_space_tuning_correct():
    """ Tests PipelineTuner for time series forecasting task with GLM model that has a complex glm search space"""
    train_data, test_data = get_ts_data(n_steps=200, forecast_length=5)

    glm_pipeline = Pipeline(PrimaryNode('glm'))
    glm_custom_params = glm_pipeline.nodes[0].custom_params
    tuned_glm_pipeline = glm_pipeline.fine_tune_all_nodes(input_data=train_data,
                                                          loss_function=mse)
    new_custom_params = tuned_glm_pipeline.nodes[0].custom_params
    assert glm_custom_params == new_custom_params


def test_greater_is_better():
    """ Tests _greater_is_better function correctness on quality metrics maximization / minimization definition"""
    target = np.array([1, 0, 1, 0, 1])
    multi_target = np.array([2, 0, 1, 0, 1, 2])
    data_type = DataTypesEnum.table
    assert _greater_is_better(target, acc, None, data_type)
    assert _greater_is_better(target, roc, None, data_type)
    assert _greater_is_better(multi_target, roc, {'multi_class': 'ovo'}, data_type)
    assert _greater_is_better(target, custom_maximized_metrics, None, data_type)
    assert not _greater_is_better(target, mse, None, data_type)
    assert not _greater_is_better(target, custom_minimized_metrics, None, data_type)
