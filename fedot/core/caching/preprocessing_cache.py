from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.preprocessing_cache_db import PreprocessingCacheDB
from fedot.core.data.data import InputData, data_type_is_table
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(BaseCache):
    """
    Usage of that class makes sense only if you have table data.
    Stores/loads DataPreprocessor's encoders and imputers for pipelines to decrease optional preprocessing time.

    :param db_path: optional str determining a file name for caching preprocessors items.
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(PreprocessingCacheDB(db_path))

    @contextmanager
    def _using_cache(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        """
        :param pipeline: pipeline to use cache for
        :param input_data: data that are going to be passed through pipeline
        """
        encoder, imputer = self.try_find_preprocessor(pipeline, input_data)
        pipeline.preprocessor.features_encoders = encoder
        pipeline.preprocessor.features_imputers = imputer
        yield
        self.add_preprocessor(pipeline, input_data)

    @staticmethod
    def manage(cache: Optional['PreprocessingCache'], pipeline: 'Pipeline',
               input_data: Union[InputData, MultiModalData]):
        """
        Gets context manager for using preprocessing cache if present or returns nullcontext otherwise.

        :param cache: preprocessors cache instance or None
        :param pipeline: pipeline to use cache for
        :param input_data: data that are going to be passed through pipeline
        """
        if (cache is None or
                isinstance(input_data, InputData) and not data_type_is_table(input_data) or
                isinstance(input_data, MultiModalData) and not any(data_type_is_table(x) for x in input_data.values())):
            return nullcontext()
        return PreprocessingCache._using_cache(cache, pipeline, input_data)

    def try_find_preprocessor(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]) -> Tuple[
        Dict[str, OneHotEncodingImplementation], Dict[str, ImputationImplementation]
    ]:
        """
        Tries to find preprocessor in DB table or returns initial otherwise.

        :param pipeline: pipeline to find preprocessor for
        :param input_data: data that are going to be passed through pipeline

        :return encoder: loaded one-hot encoder if included in DB or initial otherwise
        :return imputer: loaded imputer if included in DB or initial otherwise
        """
        try:
            structural_id = _get_db_uid(pipeline, input_data)
            processors = self._db.get_preprocessor(structural_id)
            if processors is None:
                encoder = pipeline.preprocessor.features_encoders
                imputer = pipeline.preprocessor.features_imputers
            else:
                encoder, imputer = processors
        except Exception as exc:
            self.log.warning(f'Preprocessor search error: {exc}')
        return encoder, imputer

    def add_preprocessor(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        """
        Adds preprocessor into DB working table.

        :param pipeline: pipeline with preprocessor to add
        :param input_data: data that are going to be passed through pipeline
        """
        structural_id = _get_db_uid(pipeline, input_data)
        self._db.add_preprocessor(structural_id, pipeline.preprocessor)


def _get_db_uid(pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]) -> str:
    """
    Constructs unique id from pipeline and data, which is considered as primary key for DB.

    :param pipeline: pipeline to get uid from
    :param input_data: data that are going to be passed through pipeline

    :return: unique pipeline plus related data identificator
    """
    pipeline_id = pipeline.root_node.descriptive_id
    if isinstance(input_data, InputData):
        data_id = f'{input_data.idx[0]}_{input_data.idx[-1]}'
    else:
        data_id = ':'.join([f'{x.idx[0]}_{x.idx[-1]}' for x in input_data.values()])
    return f'{pipeline_id}_{data_id}'
