import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.base_preprocessing import BasePreprocessor
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME, PipelineStructureExplorer


class _FakePreprocessor(BasePreprocessor):
    def obligatory_prepare_for_fit(self, data):
        return data

    def obligatory_prepare_for_predict(self, data):
        return data

    def optional_prepare_for_fit(self, pipeline, data):
        return data

    def optional_prepare_for_predict(self, pipeline, data):
        return data

    def label_encoding_for_fit(self, data, source_name='default'):
        return None

    def cut_dataset(self, data, border: int):
        return None

    def apply_inverse_target_encoding(self, column_to_transform):
        return column_to_transform

    def convert_indexes_for_fit(self, pipeline, data):
        return data

    def convert_indexes_for_predict(self, pipeline, data):
        return data

    def restore_index(self, input_data, result):
        return result

    def update_indices_for_time_series(self, test_data):
        return test_data

    def reduce_memory_size(self, data):
        return data


def _make_input_data(*, is_main_target=True):
    return InputData(
        idx=np.array([0, 1]),
        features=np.array([[1.0], [2.0]]),
        target=np.array([[0.0], [1.0]]),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(is_main_target=is_main_target),
    )


def _make_optional_input_data():
    data = InputData(
        idx=np.array([0, 1]),
        features=np.array([[1.0, 'a'], [np.nan, 'b']], dtype=object),
        target=np.array([[0.0], [1.0]]),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        supplementary_data=SupplementaryData(),
    )
    data.categorical_idx = np.array([1])
    data.numerical_idx = np.array([0])
    return data


def test_mark_as_preprocessed_marks_unimodal_and_multimodal_inputs():
    input_data = _make_input_data()
    multi_data = MultiModalData({'main': _make_input_data(), 'side': _make_input_data(is_main_target=False)})

    BasePreprocessor.mark_as_preprocessed(input_data)
    BasePreprocessor.mark_as_preprocessed(multi_data, is_obligatory=False)

    assert input_data.supplementary_data.obligatorily_preprocessed is True
    assert multi_data['main'].supplementary_data.optionally_preprocessed is True
    assert multi_data['side'].supplementary_data.optionally_preprocessed is True


def test_merge_preprocessors_uses_typed_merge_plan():
    api_preprocessor = _FakePreprocessor()
    pipeline_preprocessor = _FakePreprocessor()
    pipeline_preprocessor.features_encoders = {'encoder': object()}
    pipeline_preprocessor.features_imputers = {'imputer': object()}

    merged = BasePreprocessor.merge_preprocessors(
        api_preprocessor=api_preprocessor,
        pipeline_preprocessor=pipeline_preprocessor,
        use_auto_preprocessing=False,
    )

    assert merged.features_encoders == pipeline_preprocessor.features_encoders
    assert merged.features_imputers == pipeline_preprocessor.features_imputers


def test_data_preprocessor_initialization_uses_source_and_target_rules():
    preprocessor = DataPreprocessor()
    multi_data = MultiModalData({
        'main': _make_input_data(is_main_target=True),
        'side': _make_input_data(is_main_target=False),
    })

    preprocessor._init_supplementary_preprocessors(multi_data)
    preprocessor._init_main_target_source_name(multi_data)

    assert set(preprocessor.binary_categorical_processors.keys()) == {'main', 'side'}
    assert set(preprocessor.types_correctors.keys()) == {'main', 'side'}
    assert preprocessor.main_target_source_name == 'main'


def test_prepare_optional_uses_typed_optional_plan_and_target_source_resolution():
    preprocessor = DataPreprocessor()
    data = _make_optional_input_data()
    applied_steps = []

    original_check_structure = PipelineStructureExplorer.check_structure_by_tag
    preprocessor._apply_imputation_unidata = lambda current_data, source_name: applied_steps.append(
        ('imputation', source_name)
    ) or current_data
    preprocessor._apply_categorical_encoding = lambda current_data, source_name: applied_steps.append(
        ('encoding', source_name)
    ) or current_data
    PipelineStructureExplorer.check_structure_by_tag = staticmethod(
        lambda pipeline, tag_to_check, source_name: tag_to_check == 'imputation'
    )

    try:
        preprocessor._prepare_optional(object(), data, DEFAULT_SOURCE_NAME)
        preprocessor.main_target_source_name = None
        assert preprocessor._determine_target_converter() == DEFAULT_SOURCE_NAME
    finally:
        PipelineStructureExplorer.check_structure_by_tag = original_check_structure

    assert applied_steps == [('encoding', DEFAULT_SOURCE_NAME)]
