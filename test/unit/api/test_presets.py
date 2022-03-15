import datetime

from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.presets import OperationsPreset, update_builder
from fedot.core.constants import BEST_QUALITY_PRESET_NAME, \
    FAST_TRAIN_PRESET_NAME, AUTO_PRESET_NAME
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.composer.test_builder import prepare_builder_with_custom_params


def test_presets_classification():
    task = Task(TaskTypesEnum.classification)
    class_operations = get_operations_for_task(task=task, mode='all')

    excluded_tree = ['xgboost', 'catboost', 'xgbreg', 'catboostreg']
    filtered_operations = set(class_operations).difference(set(excluded_tree))
    available_operations = list(filtered_operations)

    preset_best_quality = OperationsPreset(task=task, preset_name=BEST_QUALITY_PRESET_NAME)
    operations_for_best_quality = preset_best_quality._filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name=FAST_TRAIN_PRESET_NAME)
    operations_for_fast_train = preset_fast_train._filter_operations_by_preset()

    assert len(operations_for_fast_train) < len(operations_for_best_quality) == len(available_operations)
    assert {'dt', 'logit', 'knn'} <= set(operations_for_fast_train)


def test_presets_regression():
    task = Task(TaskTypesEnum.regression)

    regr_operations = get_operations_for_task(task=task, mode='all')

    preset_best_quality = OperationsPreset(task=task, preset_name=BEST_QUALITY_PRESET_NAME)
    operations_for_best_quality = preset_best_quality._filter_operations_by_preset()

    preset_fast_train = OperationsPreset(task=task, preset_name=FAST_TRAIN_PRESET_NAME)
    operations_for_fast_train = preset_fast_train._filter_operations_by_preset()

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

    preset_best_quality = OperationsPreset(task=task, preset_name=BEST_QUALITY_PRESET_NAME)
    updated_params = preset_best_quality.composer_params_based_on_preset(composer_params)
    updated_candidates = updated_params.get('available_operations')

    assert source_candidates is None
    assert updated_candidates is not None


def test_auto_preset_processed_correctly():
    """ 'auto' preset must be replaced with appropriate preset based on initial pipeline fitting time """
    builder = prepare_builder_with_custom_params(return_all=False)
    source_available_operations = builder._composer.composer_requirements.primary
    builder = update_builder(builder=builder,
                             composer_requirements=builder._composer.composer_requirements,
                             fit_time=datetime.timedelta(minutes=0.5),
                             full_minutes_timeout=2,
                             preset=AUTO_PRESET_NAME)
    new_available_operations = builder._composer.composer_requirements.primary

    assert len(source_available_operations) != len(new_available_operations)
