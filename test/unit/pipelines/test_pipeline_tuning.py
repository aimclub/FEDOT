import os
from time import time
from random import seed

import numpy as np
import pytest
from hyperopt import hp, tpe, rand
from hyperopt.pyll.stochastic import sample as hp_sample
from sklearn.metrics import mean_squared_error as mse, accuracy_score as acc

from fedot.core.composer.metrics import ROCAUC, RMSE, Accuracy, MSE
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    GLMImplementation
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.tuner_interface import _greater_is_better
from fedot.core.validation.tune.cv_prediction import calculate_loss_function
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.tasks.test_forecasting import get_ts_data

seed(1)
np.random.seed(1)


@pytest.fixture()
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'simple_regression_train.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'simple_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


@pytest.fixture()
def tiny_classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'tiny_simple_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


@pytest.fixture()
def multi_classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../../data', 'multiclass_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


def get_simple_regr_pipeline(operation_type='rfr'):
    final = PrimaryNode(operation_type=operation_type)
    pipeline = Pipeline(final)

    return pipeline


def get_complex_regr_pipeline():
    node_scaling = PrimaryNode(operation_type='scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_linear = SecondaryNode('linear', nodes_from=[node_scaling])
    final = SecondaryNode('rfr', nodes_from=[node_ridge, node_linear])
    pipeline = Pipeline(final)

    return pipeline


def get_regr_pipelines():
    simple_pipelines = [get_simple_regr_pipeline(operation_type) for operation_type in get_regr_operation_types()]

    return simple_pipelines + [get_complex_regr_pipeline()]


def get_simple_class_pipeline(operation_type='logit'):
    final = PrimaryNode(operation_type=operation_type)
    pipeline = Pipeline(final)

    return pipeline


def get_complex_class_pipeline():
    first = PrimaryNode(operation_type='knn')
    second = PrimaryNode(operation_type='pca')
    final = SecondaryNode(operation_type='logit',
                          nodes_from=[first, second])

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
    return [RMSE.metric]


def get_class_losses():
    return [ROCAUC.metric, Accuracy.metric]


def get_not_default_search_space():
    custom_search_space = {
        'logit': {
            'C': (hp.uniform, [0.01, 5.0])
        },
        'ridge': {
            'alpha': (hp.uniform, [0.01, 5.0])
        },
        'lgbmreg': {
            'min_samples_leaf': (hp.uniform, [1e-3, 0.5]),
            'n_estimators': (hp.choice, [[100]]),
            'max_depth': (hp.choice, [[2.5, 3.5, 4.5]]),
            'learning_rate': (hp.choice, [[1e-3, 1e-2, 1e-1]]),
            'subsample': (hp.uniform, [0.15, 1])
        },
        'dt': {
            'min_samples_leaf': (hp.uniform, [1e-3, 0.5]),
            'max_depth': (hp.choice, [[-1]])
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


def custom_maximized_metrics(real_data: InputData, pred_data: OutputData):
    mse_value = mse(real_data.target, pred_data.predict, squared=False)
    return -(mse_value + 2) * 0.5


def custom_minimized_metrics(real_data: InputData, pred_data: OutputData):
    acc_value = acc(real_data.target, pred_data.predict)
    return 100 - (acc_value + 2) * 0.5


def run_pipeline_tuner(train_data,
                       pipeline,
                       loss_function,
                       search_space=SearchSpace(),
                       cv=None,
                       algo=tpe.suggest,
                       iterations=1,
                       early_stopping_rounds=None):
    # Pipeline tuning
    pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                   task=train_data.task,
                                   iterations=iterations,
                                   early_stopping_rounds=early_stopping_rounds,
                                   search_space=search_space,
                                   algo=algo)
    _ = pipeline_tuner.tune_pipeline(input_data=train_data,
                                     cv_folds=cv,
                                     loss_function=loss_function)
    return pipeline_tuner


def run_sequential_tuner(train_data,
                         pipeline,
                         loss_function,
                         search_space=SearchSpace(),
                         cv=None,
                         algo=tpe.suggest,
                         iterations=1,
                         early_stopping_rounds=None):
    # Pipeline tuning
    sequential_tuner = SequentialTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=iterations,
                                       early_stopping_rounds=early_stopping_rounds,
                                       search_space=search_space,
                                       algo=algo)
    # Optimization will be performed on RMSE metric, so loss params are defined
    _ = sequential_tuner.tune_pipeline(input_data=train_data,
                                       cv_folds=cv,
                                       loss_function=loss_function)
    return sequential_tuner


def run_node_tuner(train_data,
                   pipeline,
                   loss_function,
                   search_space=SearchSpace(),
                   cv=None,
                   node_index=0,
                   algo=tpe.suggest,
                   iterations=1,
                   early_stopping_rounds=None):
    # Pipeline tuning
    node_tuner = SequentialTuner(pipeline=pipeline,
                                 task=train_data.task,
                                 iterations=iterations,
                                 early_stopping_rounds=early_stopping_rounds,
                                 search_space=search_space,
                                 algo=algo)
    _ = node_tuner.tune_node(input_data=train_data,
                             node_index=node_index,
                             cv_folds=cv,
                             loss_function=loss_function)
    return node_tuner


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    pipeline = get_complex_class_pipeline()

    custom_params = dict(C=10)

    pipeline.root_node.custom_params = custom_params
    pipeline.fit(data)
    params = pipeline.root_node.fitted_operation.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_pipeline_tuner_correct(data_fixture, pipelines, loss_functions, request):
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    cvs = [None, 2]

    for pipeline in pipelines:
        for loss_function in loss_functions:
            for cv in cvs:
                pipeline_tuner = run_pipeline_tuner(train_data=train_data,
                                                    pipeline=pipeline,
                                                    loss_function=loss_function,
                                                    cv=cv)
                assert pipeline_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_pipeline_tuner_with_custom_search_space(data_fixture, pipelines, loss_functions, request):
    """ Test PipelineTuner with different search spaces """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    search_spaces = [SearchSpace(), get_not_default_search_space()]

    for search_space in search_spaces:
        pipeline_tuner = run_pipeline_tuner(train_data=train_data,
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
def test_sequential_tuner_correct(data_fixture, pipelines, loss_functions, request):
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    cvs = [None, 2]

    for pipeline in pipelines:
        for loss_function in loss_functions:
            for cv in cvs:
                sequential_tuner = run_sequential_tuner(train_data=train_data,
                                                        pipeline=pipeline,
                                                        loss_function=loss_function,
                                                        cv=cv)
                assert sequential_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_sequential_tuner_with_custom_search_space(data_fixture, pipelines, loss_functions, request):
    """ Test SequentialTuner with different search spaces """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    search_spaces = [SearchSpace(), get_not_default_search_space()]

    for search_space in search_spaces:
        sequential_tuner = run_sequential_tuner(train_data=train_data,
                                                pipeline=pipelines[0],
                                                loss_function=loss_functions[0],
                                                search_space=search_space)
        assert sequential_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture, pipelines, loss_functions',
                         [('regression_dataset', get_regr_pipelines(), get_regr_losses()),
                          ('classification_dataset', get_class_pipelines(), get_class_losses()),
                          ('multi_classification_dataset', get_class_pipelines(), get_class_losses())])
def test_certain_node_tuning_correct(data_fixture, pipelines, loss_functions, request):
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    cvs = [None, 2]

    for pipeline in pipelines:
        for loss_function in loss_functions:
            for cv in cvs:
                node_tuner = run_node_tuner(train_data=train_data,
                                            pipeline=pipeline,
                                            loss_function=loss_function,
                                            cv=cv)
                assert node_tuner.obtained_metric is not None

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
    search_spaces = [SearchSpace(), get_not_default_search_space()]

    for search_space in search_spaces:
        node_tuner = run_node_tuner(train_data=train_data,
                                    pipeline=pipelines[0],
                                    loss_function=loss_functions[0],
                                    search_space=search_space)
        assert node_tuner.obtained_metric is not None

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('n_steps', [100, 133, 217, 300])
def test_ts_pipeline_with_stats_model(n_steps):
    """ Tests PipelineTuner for time series forecasting task with AR model """
    train_data, test_data = get_ts_data(n_steps=n_steps, forecast_length=5)

    ar_pipeline = Pipeline(PrimaryNode('ar'))

    for search_space in [SearchSpace(), get_not_default_search_space()]:
        # Tune AR model
        tuner_ar = PipelineTuner(pipeline=ar_pipeline, task=train_data.task, iterations=3,
                                 search_space=search_space, algo=rand.suggest)
        tuned_ar_pipeline = tuner_ar.tune_pipeline(input_data=train_data,
                                                   loss_function=MSE.metric)

    is_tuning_finished = True

    assert is_tuning_finished


@pytest.mark.parametrize('data_fixture', ['tiny_classification_dataset'])
def test_early_stop_in_tuning(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    start_pipeline_tuner = time()
    _ = run_pipeline_tuner(train_data=train_data,
                           pipeline=get_class_pipelines()[0],
                           loss_function=ROCAUC.metric,
                           iterations=1000,
                           early_stopping_rounds=1)
    assert time() - start_pipeline_tuner < 1

    start_sequential_tuner = time()
    _ = run_sequential_tuner(train_data=train_data,
                             pipeline=get_class_pipelines()[0],
                             loss_function=ROCAUC.metric,
                             iterations=1000,
                             early_stopping_rounds=1)
    assert time() - start_sequential_tuner < 1

    start_node_tuner = time()
    _ = run_node_tuner(train_data=train_data,
                       pipeline=get_class_pipelines()[0],
                       loss_function=ROCAUC.metric,
                       iterations=1000,
                       early_stopping_rounds=1)
    assert time() - start_node_tuner < 1


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
                                                          loss_function=MSE.metric)
    new_custom_params = tuned_glm_pipeline.nodes[0].custom_params
    assert glm_custom_params == new_custom_params


def test_greater_is_better():
    """ Tests _greater_is_better function correctness on quality metrics maximization / minimization definition"""

    assert _greater_is_better(custom_maximized_metrics)
    assert not _greater_is_better(MSE.metric)
    assert not _greater_is_better(custom_minimized_metrics)
    # these metrics are given with minus
    assert not _greater_is_better(Accuracy.metric)
    assert not _greater_is_better(ROCAUC.metric)


def test_calculate_loss_function_for_classification_label():
    """ Tests _calculate_loss_function correctness on quality metrics"""

    target = InputData(features=np.arange(5),
                       idx=np.arange(5),
                       target=np.array([1, 0, 1, 0, 1]),
                       task=Task(TaskTypesEnum.classification),
                       data_type=DataTypesEnum.table)

    multi_target = InputData(features=np.arange(5),
                             idx=np.arange(5),
                             target=np.array([2, 0, 1, 0, 1]),
                             task=Task(TaskTypesEnum.classification),
                             data_type=DataTypesEnum.table)

    pred_clear = OutputData(features=np.arange(5),
                            idx=np.arange(5),
                            predict=np.array([1, 0, 1, 0, 0]),
                            task=Task(TaskTypesEnum.classification),
                            data_type=DataTypesEnum.table)

    multi_pred_clear = OutputData(features=np.arange(5),
                                  idx=np.arange(5),
                                  predict=np.array([2, 0, 1, 0, 2]),
                                  task=Task(TaskTypesEnum.classification),
                                  data_type=DataTypesEnum.table)

    assert np.isclose(calculate_loss_function(Accuracy.metric, target, pred_clear), -0.8)
    assert np.isclose(calculate_loss_function(Accuracy.metric, multi_target, multi_pred_clear), -0.8)


def test_calculate_loss_function_for_classification_proba():
    target = InputData(features=np.arange(5),
                       idx=np.arange(5),
                       target=np.array([1, 0, 1, 0, 1]),
                       task=Task(TaskTypesEnum.classification),
                       data_type=DataTypesEnum.table)

    multi_target = InputData(features=np.arange(5),
                             idx=np.arange(5),
                             target=np.array([2, 0, 1, 0, 1]),
                             task=Task(TaskTypesEnum.classification),
                             data_type=DataTypesEnum.table)

    pred_prob = OutputData(features=np.arange(5),
                           idx=np.arange(5),
                           predict=np.array([0.8, 0.3, 0.6, 0.49, 0.49]),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    multi_pred_prob = OutputData(features=np.arange(5),
                                 idx=np.arange(5),
                                 predict=np.array([[0.2, 0.3, 0.5],
                                                   [0.6, 0.3, 0.1],
                                                   [0.3, 0.4, 0.3],
                                                   [0.5, 0.4, 0.1],
                                                   [0.1, 0.4, 0.5]]),
                                 task=Task(TaskTypesEnum.classification),
                                 data_type=DataTypesEnum.table)

    assert np.isclose(calculate_loss_function(Accuracy.metric, target, pred_prob), -0.8)
    assert np.isclose(calculate_loss_function(ROCAUC.metric, target, pred_prob), -0.917)

    assert np.isclose(calculate_loss_function(Accuracy.metric, multi_target, multi_pred_prob), -0.8)
    assert np.isclose(calculate_loss_function(ROCAUC.metric, multi_target, multi_pred_prob), -0.903)


def test_calculate_loss_function_for_regression():
    regr_target = InputData(features=np.arange(5),
                            idx=np.arange(5),
                            target=np.array([0.2, 0.1, 1, 0.3, 1.7]),
                            task=Task(TaskTypesEnum.classification),
                            data_type=DataTypesEnum.table)

    regr_pred = OutputData(features=np.arange(5),
                           idx=np.arange(5),
                           predict=np.array([0.23, 0.15, 1.2, 0.4, 1.16]),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    assert np.isclose(calculate_loss_function(MSE.metric, regr_target, regr_pred), 0.069)
