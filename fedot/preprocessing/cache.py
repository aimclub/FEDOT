from typing import TYPE_CHECKING, Optional, Union

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, SingletonMeta, default_log
from fedot.preprocessing.cache_db import PreprocessingCacheDB

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(metaclass=SingletonMeta):
    def __init__(self, log: Optional[Log] = None, db_path: Optional[str] = None):
        self.log = log or default_log(__name__)
        self._db = PreprocessingCacheDB(db_path)

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


def _get_pipeline_structural_id(pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]) -> str:
    pipeline_id = pipeline.root_node.descriptive_id
    # if isinstance(input_data, InputData):
    #     data_id = ''.join(str(input_data.features[[0, -1]]))
    # else:
    #     data_id = ''.join([str(x.features[[0, -1]]) for x in input_data.values()])
    # return f'{pipeline_id}_{data_id}'  # re.sub(f'[{string.punctuation}]+', '', pipeline.root_node.descriptive_id)
    return pipeline_id
