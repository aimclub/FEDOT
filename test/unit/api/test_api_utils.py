from copy import deepcopy

import numpy as np

from examples.simple.classification.classification_pipelines import classification_pipeline_without_balancing
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.initial_assumptions import ApiInitialAssumptions
from fedot.api.main import Fedot
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.preprocessing import DataPreprocessor
from ..api.test_main_api import get_dataset
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

from testfixtures import LogCapture


def test_compose_fedot_model_with_tuning():
    api_composer = ApiComposer('classification')
    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)

    task = Task(task_type=TaskTypesEnum.classification)
    generations = 1

    with LogCapture() as logs:
        _, _, history = api_composer.compose_fedot_model(api_params=dict(train_data=train_input,
                                                                         task=task,
                                                                         logger=default_log('test_log'),
                                                                         timeout=0.1,
                                                                         n_jobs=1,
                                                                         initial_assumption=None),
                                                         composer_params=dict(max_depth=1,
                                                                              max_arity=1,
                                                                              pop_size=2,
                                                                              num_of_generations=generations,
                                                                              available_operations=None,
                                                                              composer_metric=None,
                                                                              validation_blocks=None,
                                                                              cv_folds=None,
                                                                              genetic_scheme=None),
                                                         tuning_params=dict(with_tuning=True,
                                                                            tuner_metric=None))
    expected = ('test_log', 'INFO', 'Composed pipeline returned without tuning.')
    logs.check_present(expected, order_matters=False)


def test_output_classification_converting_correct():
    """ Check the correctness of correct prediction method for binary classification task """

    task = Task(task_type=TaskTypesEnum.classification)
    real = InputData(idx=[0, 1, 2], features=np.array([[0], [1], [2]]),
                     target=np.array([[0], [1], [1]]),
                     task=task, data_type=DataTypesEnum.table)
    prediction = OutputData(idx=[0, 1, 2], features=np.array([[0], [1], [2]]),
                            predict=np.array([[0.5], [0.7], [0.7]]),
                            target=np.array([[0], [1], [1]]),
                            task=task, data_type=DataTypesEnum.table)

    data_processor = ApiDataProcessor(task=task)
    # Perform correction inplace
    data_processor.correct_predictions(metric_name='f1', real=real, prediction=prediction)

    assert int(prediction.predict[1]) == 1


def test_predefined_initial_assumption():
    """ Check if predefined initial assumption and other api params don't lose while preprocessing is performing"""
    train_input, _, _ = get_dataset(task_type='classification')
    initial_pipeline = classification_pipeline_without_balancing()
    available_operations = ['bernb', 'dt', 'knn', 'lda', 'qda', 'logit', 'rf', 'svc',
                            'scaling', 'normalization', 'pca', 'kernel_pca']

    model = Fedot(problem='classification', timeout=1,
                  verbose_level=4, composer_params={'timeout': 1,
                                                    'available_operations': available_operations,
                                                    'initial_assumption': initial_pipeline})
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
    logger = default_log('FEDOT logger', verbose_level=4)
    available_operations = ['dt']

    api_initial_assumption = ApiInitialAssumptions()
    initial_assumption = \
        api_initial_assumption.get_initial_assumption(data=train_input, task=Task(TaskTypesEnum.classification),
                                                      available_operations=available_operations,
                                                      logger=logger)
    res_init_assumption = Pipeline(PrimaryNode('dt'))
    assert initial_assumption[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_init_assumption_with_inappropriate_available_operations():
    """ Checks that if given available operations are not suitable for the task,
    then the default initial assumption will be formed """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    logger = default_log('FEDOT logger', verbose_level=4)
    available_operations = ['linear', 'xgboost', 'lagged']

    api_initial_assumption = ApiInitialAssumptions()
    initial_assumption = \
        api_initial_assumption.get_initial_assumption(data=train_input, task=Task(TaskTypesEnum.classification),
                                                      available_operations=available_operations,
                                                      logger=logger)
    primary = PrimaryNode('scaling')
    root = SecondaryNode('rf', nodes_from=[primary])
    res_init_assumption = Pipeline(root)

    assert initial_assumption[0].root_node.descriptive_id == res_init_assumption.root_node.descriptive_id


def test_api_composer_divide_operations():
    """ Checks whether the composer correctly divides operations into primary and secondary """

    available_operations = ['logit', 'rf', 'dt', 'xgboost']

    api_composer = ApiComposer(problem='classification')
    primary, secondary = \
        api_composer.divide_operations(task=Task(TaskTypesEnum.classification),
                                       available_operations=available_operations)

    assert primary == available_operations
    assert secondary == available_operations
