from typing import Tuple, Callable

import pandas as pd
import pytest

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.preprocessing.dummy_preprocessing import DummyPreprocessor
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column, \
    data_with_only_categorical_features


def test_pipeline_has_dummy_preprocessor_with_disabled_preprocessing():
    """
    Tests pipeline with disabled input data preprocessing has dummy preprocessor
    """
    pipeline = Pipeline(PrimaryNode('ridge'), use_input_preprocessing=False)
    assert type(pipeline.preprocessor) is DummyPreprocessor


def _assert_equal_data(data1: InputData, data2: InputData):
    assert data1.features.shape == data2.features.shape
    assert data1.target.shape == data2.target.shape

    assert pd.DataFrame(data1.features).equals(pd.DataFrame(data2.features))
    assert pd.DataFrame(data1.target).equals(pd.DataFrame(data2.target))


@pytest.mark.parametrize('case', [
    (data_with_mixed_types_in_each_column, True),
    (data_with_mixed_types_in_each_column, False)
])
def test_disabled_pipeline_data_preprocessing(case: Tuple[Callable[[], InputData], bool]):
    """
    Tests pipeline with disabled input data preprocessing preprocesses the data in no way

    Args:
        case: input_data for preprocessing and is it preprocessing for fit stage
    """
    data_getter, is_fit_stage = case
    input_data = data_getter()

    pipeline = Pipeline(PrimaryNode('ridge'), use_input_preprocessing=False)
    if is_fit_stage:
        preprocessed_data = pipeline._preprocess(input_data, is_fit_stage=is_fit_stage)
    else:
        try:
            preprocessed_data = pipeline._preprocess(input_data, is_fit_stage=is_fit_stage)
        except Exception as exc:
            assert False, 'Raised exception during predict preprocessing even without fit stage'
    assert id(preprocessed_data) != id(input_data)
    _assert_equal_data(preprocessed_data, input_data)


# TODO: check _postprocessing of prediction output

def test_data_preprocessor_inits_supplementary_preprocessors_only_once():
    input_data = data_with_only_categorical_features()

    preprocessor = DataPreprocessor()

    preprocessor._init_supplementary_preprocessors(input_data)
    prev = [
        {k: id(v) for k, v in preprocessor.binary_categorical_processors.items()},
        {k: id(v) for k, v in preprocessor.types_correctors.items()}
    ]
    preprocessor._init_supplementary_preprocessors(input_data)
    cur = [
        {k: id(v) for k, v in preprocessor.binary_categorical_processors.items()},
        {k: id(v) for k, v in preprocessor.types_correctors.items()}
    ]
    are_equal = [
        dct1 == dct2
        for dct1, dct2 in zip(prev, cur)
    ]
    assert are_equal[0]
    assert are_equal[1]


def test_data_preprocessor_performs_obligatory_data_preprocessing_only_once():
    input_data = data_with_only_categorical_features()

    preprocessor = DataPreprocessor()

    preprocessed_data = preprocessor.obligatory_prepare_for_fit(input_data)
    preprocessed_data_same = preprocessor.obligatory_prepare_for_fit(preprocessed_data)

    assert id(preprocessed_data) == id(preprocessed_data_same)
    _assert_equal_data(preprocessed_data, preprocessed_data_same)


def test_data_preprocessor_performs_optional_data_preprocessing_only_once():
    input_data = data_with_only_categorical_features()
    preprocessor = DataPreprocessor()
    pipeline = Pipeline(PrimaryNode('ridge'))

    preprocessed_data = preprocessor.optional_prepare_for_fit(pipeline, input_data)

    other_pipeline = Pipeline(PrimaryNode('dt'))
    preprocessed_data_same = preprocessor.optional_prepare_for_fit(other_pipeline, preprocessed_data)

    assert id(preprocessed_data) == id(preprocessed_data_same)
    _assert_equal_data(preprocessed_data, preprocessed_data_same)
