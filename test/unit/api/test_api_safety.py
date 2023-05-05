import numpy as np

from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.input_analyser import InputAnalyser
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.integration.api.test_main_api import default_params


def get_data_analyser_with_specific_params(max_size=18, max_cat_cardinality=5):
    """ Create a DataAnalyser object with small max dataset size and small max cardinality for categorical features"""
    safety_module = InputAnalyser(safe_mode=True)
    preprocessor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    safety_module.max_size = max_size
    safety_module.max_cat_cardinality = max_cat_cardinality
    return safety_module, preprocessor


def get_small_cat_data():
    """ Generate tabular data with categorical features."""
    features = np.array([["a", "qq", 0.5],
                         ["b", "pp", 1],
                         ["c", np.nan, 3],
                         ["d", "oo", 3],
                         ["d", "oo", 3],
                         ["d", "oo", 3],
                         ["d", "oo", 3],
                         ["d", "oo", 3]], dtype=object)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    input_data = InputData(idx=np.arange(features.shape[0]),
                           features=features, target=target,
                           data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    input_data = DataPreprocessor().obligatory_prepare_for_fit(input_data)
    return input_data


def test_safety_label_correct():
    """
    Check if cutting and label encoding is used for  pseudo large data with categorical features with high cardinality
    """
    api_safety, api_preprocessor = get_data_analyser_with_specific_params()
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)
    assert data.features.shape[0] * data.features.shape[1] <= api_safety.max_size
    assert data.features.shape[1] == 3
    assert data.features[0, 0] != 'a'


def test_recommendations_works_correct_in_final_fit():
    """
    Check if accept and apply recommendations works correct with new data object
    """

    api_safety, api_preprocessor = get_data_analyser_with_specific_params()
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)

    data_new = get_small_cat_data()
    if recs_for_data:
        # if data was cut we need to refit pipeline on full data
        api_preprocessor.accept_and_apply_recommendations(data_new,
                                                          {k: v for k, v in recs_for_data.items()
                                                           if k != 'cut'})

    assert data_new.features.shape[1] == 3
    assert data_new.features[0, 0] != 'a'


def test_no_safety_needed_correct():
    """
    Check if oneHot encoding is used for small data with small cardinality of categorical features
    """
    api_safety, api_preprocessor = get_data_analyser_with_specific_params(max_size=100, max_cat_cardinality=100)
    data = get_small_cat_data()
    recs_for_data, _ = api_safety.give_recommendations(data)
    api_preprocessor.accept_and_apply_recommendations(data, recs_for_data)
    assert data.features.shape[0] * data.features.shape[1] == 24
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == 'a'


def test_api_fit_predict_with_pseudo_large_dataset_with_label_correct():
    """
    Test if safe mode in API cut large data and use LabelEncoder for features with high cardinality
    """
    model = Fedot(problem='classification',
                  preset='fast_train',
                  safe_mode=True)
    model.data_analyser.max_cat_cardinality = 5
    model.data_analyser.max_size = 18
    data = get_small_cat_data()
    pipeline = model.fit(features=data, predefined_model='auto')
    pipeline.predict(data)
    model.predict(features=data)

    # there should be only tree like models + data operations
    assert len(model.params.get('available_operations')) == 5
    assert 'logit' not in model.params.get('available_operations')


def test_api_fit_predict_with_pseudo_large_dataset_with_onehot_correct():
    """
    Test if safe mode in API use OneHotEncoder with small data with small cardinality
    """
    model = Fedot(problem="classification", **default_params)
    model.data_analyser.max_size = 1000
    data = get_small_cat_data()
    model.fit(features=data, predefined_model='auto')

    model.predict(features=data)
    # there should be all light models + data operations
    assert 'logit' in model.params.get('available_operations')
