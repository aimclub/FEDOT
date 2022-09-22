import logging
import pathlib
import random
from copy import deepcopy

from examples.simple.classification.classification_pipelines import classification_pipeline_without_balancing
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.assumptions.assumptions_builder import AssumptionsBuilder
from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import default_fedot_data_dir
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.unit.api.test_main_api import get_dataset
from test.unit.tasks.test_classification import get_binary_classification_data


def test_compose_fedot_model_with_tuning():
    test_file_pth = pathlib.Path(default_fedot_data_dir(), 'test_logger_file.log')
    log = Log(log_file=test_file_pth, output_logging_level=logging.DEBUG)
    logger = log.get_adapter('test_log')

    api_composer = ApiComposer('classification')
    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)

    task = Task(task_type=TaskTypesEnum.classification)
    operations = get_operations_for_task(task=task, mode='model')
    generations = 1

    api_composer.compose_fedot_model(api_params=dict(train_data=train_input,
                                                     task=task,
                                                     logger=logger,
                                                     timeout=0.1,
                                                     n_jobs=1,
                                                     show_progress=False),
                                     composer_params=dict(max_depth=1,
                                                          max_arity=1,
                                                          pop_size=2,
                                                          num_of_generations=generations,
                                                          keep_n_best=1,
                                                          available_operations=operations,
                                                          metric=None,
                                                          validation_blocks=None,
                                                          cv_folds=None,
                                                          genetic_scheme=None,
                                                          max_pipeline_fit_time=None,
                                                          collect_intermediate_metric=False,
                                                          preset='fast_train',
                                                          initial_assumption=None,
                                                          use_pipelines_cache=False,
                                                          use_preprocessing_cache=False,
                                                          cache_folder=None),
                                     tuning_params=dict(with_tuning=True,
                                                        tuner_metric=None))
    with open(log.log_file, 'r') as f:
        log_text = f.read()
        assert 'Composed pipeline returned without tuning.' in log_text


def test_output_binary_classification_correct():
    """ Check the correctness of prediction for binary classification task """

    task_type = 'classification'

    data = get_binary_classification_data()

    random.seed(1)
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    model = Fedot(problem=task_type, timeout=0.1)
    model.fit(train_data, predefined_model='logit')
    model.predict(test_data)
    metrics = model.get_metrics(metric_names=['roc_auc', 'f1'])

    assert metrics['roc_auc'] >= 0.6
    assert metrics['f1'] >= 0.6


def test_predefined_initial_assumption():
    """ Check if predefined initial assumption and other api params don't lose while preprocessing is performing"""
    train_input, _, _ = get_dataset(task_type='classification')
    initial_pipeline = classification_pipeline_without_balancing()
    available_operations = ['bernb', 'dt', 'knn', 'lda', 'qda', 'logit', 'rf', 'svc',
                            'scaling', 'normalization', 'pca', 'kernel_pca']

    model = Fedot(problem='classification', timeout=1.,
                  logging_level=logging.DEBUG, available_operations=available_operations,
                  initial_assumption=initial_pipeline)
    model.target = train_input.target
    model.train_data = model.data_processor.define_data(features=train_input.features,
                                                        target=train_input.target,
                                                        is_predict=False)
    old_params = deepcopy(model.params)
    recommendations = model.data_analyser.give_recommendation(model.train_data)
    model.data_processor.accept_and_apply_recommendations(model.train_data, recommendations)
    model.params.accept_and_apply_recommendations(model.train_data, recommendations)

    assert model.params.api_params['initial_assumption'] is not None
    assert len(old_params.api_params) == len(model.params.api_params)


def test_the_formation_of_initial_assumption():
    """ Checks that the initial assumption is formed based on the given available operations """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    available_operations = ['dt']

    initial_assumptions = AssumptionsBuilder \
        .get(train_input) \
        .from_operations(available_operations) \
        .build()
    res_init_assumption = Pipeline(PrimaryNode('dt'))
    assert initial_assumptions[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_init_assumption_with_inappropriate_available_operations():
    """ Checks that if given available operations are not suitable for the task,
    then the default initial assumption will be formed """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    available_operations = ['linear', 'xgboost', 'lagged']

    initial_assumptions = AssumptionsBuilder \
        .get(train_input) \
        .from_operations(available_operations) \
        .build()
    primary = PrimaryNode('scaling')
    root = SecondaryNode('rf', nodes_from=[primary])
    res_init_assumption = Pipeline(root)

    assert initial_assumptions[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_api_composer_available_operations():
    """ Checks if available_operations goes through all fitting process"""
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=1))
    train_data, _, _ = get_dataset(task_type='ts_forecasting')
    available_operations = ['lagged']
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=0.01,
                  available_operations=available_operations,
                  pop_size=500
                  )
    model.fit(train_data)
    assert model.params.api_params['available_operations'] == available_operations
