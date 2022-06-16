from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.presets import OperationsPreset
from fedot.api.main import Fedot
from fedot.core.constants import FAST_TRAIN_PRESET_NAME
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.api.test_main_api import data_with_binary_features_and_categorical_target


def test_presets_classification():
    task = Task(TaskTypesEnum.classification)
    class_operations = get_operations_for_task(task=task, mode='all')

    excluded_tree = ['xgboost', 'catboost', 'xgbreg', 'catboostreg']
    filtered_operations = set(class_operations).difference(set(excluded_tree))
    available_operations = list(filtered_operations)

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    operations_for_best_quality = preset_best_quality.filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name='fast_train')
    operations_for_fast_train = preset_fast_train.filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) == len(available_operations)
    assert {'dt', 'logit', 'knn'} <= set(operations_for_fast_train)


def test_presets_regression():
    task = Task(TaskTypesEnum.regression)

    regr_operations = get_operations_for_task(task=task, mode='all')

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    operations_for_best_quality = preset_best_quality.filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name='fast_train')
    operations_for_fast_train = preset_fast_train.filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) <= len(regr_operations)
    assert {'dtreg', 'lasso', 'ridge', 'linear'} <= set(operations_for_fast_train)


def test_presets_inserting_in_params_correct():
    """
    Check if operations from presets are correctly included in the dictionary
    with parameters for the composer
    """
    composer_params = ApiParams.get_default_evo_params(problem='regression')
    source_candidates = composer_params.get('available_operations')

    task = Task(TaskTypesEnum.regression)

    preset_best_quality = OperationsPreset(task=task, preset_name='best_quality')
    updated_params = preset_best_quality.composer_params_based_on_preset(composer_params)
    updated_candidates = updated_params.get('available_operations')

    assert source_candidates is None
    assert updated_candidates is not None


def test_auto_preset_converted_correctly():
    """ Checks that the proposed method of automatic preset detection correctly converts a preset """
    data = data_with_binary_features_and_categorical_target()

    simple_init_assumption = Pipeline(PrimaryNode('logit'))
    fedot_model = Fedot(problem='classification', preset='auto', timeout=0.01,
                        composer_params={'initial_assumption': simple_init_assumption, 'pop_size': 500})
    # API must return initial assumption without composing and tuning (due to population size is too large)
    fedot_model.fit(data)
    assert fedot_model.api_composer.preset_name == FAST_TRAIN_PRESET_NAME
