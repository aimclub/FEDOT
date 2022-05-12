from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Tuple, Union

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, SingletonMeta, default_log
from fedot.preprocessing.cache_db import PreprocessingCacheDB
from fedot.preprocessing.structure import PipelineStructureExplorer

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(metaclass=SingletonMeta):
    """
    Stores/loads preprocessors for pipelines to decrease time for fitting preprocessor.

    :param log: optional Log object to record messages
    :param db_path: optional str db file name
    """

    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None):
        self.log = log or default_log(__name__)
        self._db = PreprocessingCacheDB(db_path)

    @contextmanager
    def using_cache(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        pipeline.preprocessor = self.try_find_preprocessor(pipeline, input_data)
        yield
        self.add_preprocessor(pipeline, input_data)

    def try_find_preprocessor(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        try:
            structural_id = _get_pipeline_structural_id(pipeline, input_data)
            matched = self._db.get_preprocessor(structural_id)
            if matched is not None:
                return matched
        except Exception as exc:
            self.log.error(f'Preprocessor search error: {exc}')
        return pipeline.preprocessor

    def add_preprocessor(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        structural_id = _get_pipeline_structural_id(pipeline, input_data)
        self._db.add_preprocessor(structural_id, pipeline.preprocessor)

    def reset(self):
        self._db.reset()


_structure_explorer = PipelineStructureExplorer()


def get_struct_info(pipeline: 'Pipeline', input_data: InputData, source_name: str) -> Tuple[bool, bool, bool, bool]:
    has_cats = data_has_categorical_features(input_data)
    has_gaps = data_has_missing_values(input_data)

    has_imputer, has_encoder = [
        _structure_explorer.check_structure_by_tag(pipeline, tag, source_name)
        for tag in ['imputation', 'encoding']
    ]
    return has_cats, has_gaps, has_imputer, has_encoder


def _get_pipeline_structural_id(pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]) -> str:
    # struct_id = ''
    # if isinstance(input_data, InputData):
    #     has_cats, has_gaps, has_imputer, has_encoder = get_struct_info(pipeline, input_data, DEFAULT_SOURCE_NAME)
    #     struct_id += f'_{has_cats}_{has_gaps}_{has_imputer}_{has_encoder}'
    # else:
    #     for data_source_name, values in input_data.items():
    #         has_cats, has_gaps, has_imputer, has_encoder = get_struct_info(pipeline, values, data_source_name)
    #         struct_id += f'_{has_cats}_{has_gaps}_{has_imputer}_{has_encoder}'
    # return struct_id
    pipeline_id = pipeline.root_node.descriptive_id
    if isinstance(input_data, InputData):
        data_id = ''.join(str(input_data.features[[0, -1]]))
    else:
        data_id = ''.join([str(x.features[[0, -1]]) for x in input_data.values()])
    return f'{pipeline_id}_{data_id}'
