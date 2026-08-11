import numpy as np

from fedot import Fedot
from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.input_analyser import InputAnalyser
from fedot.core.data.input_data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.integration.api.test_main_api import TESTS_MAIN_API_DEFAULT_PARAMS


def get_data_analyser_with_specific_params(max_size=18, max_cat_cardinality=5):
    safety_module = InputAnalyser(safe_mode=True)
    preprocessor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    safety_module.max_size = max_size
    safety_module.max_cat_cardinality = max_cat_cardinality
    return safety_module, preprocessor


def get_small_cat_data():
    features = np.array([['a', 'qq', 0.5],
                         ['b', 'pp', 1],
                         ['c', np.nan, 3],
                         ['d', 'oo', 3],
                         ['d', 'oo', 3],
                         ['d', 'oo', 3],
                         ['d', 'oo', 3],
                         ['d', 'oo', 3]], dtype=object)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return InputData(idx=np.arange(features.shape[0]),
                     features=features, target=target,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))


def test_safety_label_correct():
    # Recommendations are still produced, but applying them is a no-op.
    api_safety, api_preprocessor = get_data_analyser_with_specific_params()
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    assert 'cut' in recs_for_data
    assert 'label_encoded' in recs_for_data
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == 'a'


def test_recommendations_works_correct_in_final_fit():
    api_safety, api_preprocessor = get_data_analyser_with_specific_params()
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)

    data_new = get_small_cat_data()
    if recs_for_data:
        api_preprocessor.accept_and_apply_recommendations(
            data_new,
            {k: v for k, v in recs_for_data.items() if k != 'cut'},
        )

    assert data_new.features.shape[1] == 3
    assert data_new.features[0, 0] == 'a'


def test_no_safety_needed_correct():
    api_safety, api_preprocessor = get_data_analyser_with_specific_params(
        max_size=100, max_cat_cardinality=100)
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)
    assert recs_for_data == {}
    assert data.features.shape[0] * data.features.shape[1] == 24
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == 'a'


def test_recommendations_switch_preset_for_high_cardinality_categorical_data():
    model = Fedot(problem='classification',
                  preset='fast_train',
                  safe_mode=True)
    model.data_analyser.max_cat_cardinality = 5
    model.data_analyser.max_size = 18
    data = get_small_cat_data()

    model.params.update_available_operations_by_preset(data)
    recs_for_data, _ = model.data_analyser.give_recommendations(data)
    model.params.accept_and_apply_recommendations(data, recs_for_data)

    assert 'label_encoded' in recs_for_data
    assert len(model.params.get('available_operations')) == 5
    assert 'logit' not in model.params.get('available_operations')


def test_recommendations_keep_onehot_preset_for_low_cardinality_categorical_data():
    model = Fedot(problem='classification', **TESTS_MAIN_API_DEFAULT_PARAMS)
    model.data_analyser.max_size = 1000
    data = get_small_cat_data()

    model.params.update_available_operations_by_preset(data)
    recs_for_data, _ = model.data_analyser.give_recommendations(data)
    model.params.accept_and_apply_recommendations(data, recs_for_data)

    assert 'label_encoded' not in recs_for_data
    assert 'logit' in model.params.get('available_operations')
