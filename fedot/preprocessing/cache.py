from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from fedot.core.caching.base_cache import BaseCache
from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)
from fedot.preprocessing.cache_db import PreprocessingCacheDB
from fedot.preprocessing.structure import PipelineStructureExplorer

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(BaseCache):
    """
    Stores/loads preprocessors for pipelines to decrease time for fitting preprocessor.

    :param db_path: optional str db file name
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
    def using_cache(cache: Optional['PreprocessingCache'], pipeline: 'Pipeline',
                    input_data: Union[InputData, MultiModalData]):
        """
        Gets context manager for using preprocessing cache if present or returns nullcontext otherwise.

        :param cache: preprocessors cache instance or None
        :param pipeline: pipeline to use cache for
        :param input_data: data that are going to be passed through pipeline
        """
        if cache is None:
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
            structural_id = _get_pipeline_structural_id(pipeline, input_data)
            processors = self._db.get_preprocessor(structural_id)
            if processors is None:
                encoder = pipeline.preprocessor.features_encoders
                imputer = pipeline.preprocessor.features_imputers
            else:
                encoder, imputer = processors
        except Exception as exc:
            self.log.error(f'Preprocessor search error: {exc}')
        return encoder, imputer

    def add_preprocessor(self, pipeline: 'Pipeline', input_data: Union[InputData, MultiModalData]):
        """
        Adds preprocessor into DB working table.

        :param pipeline: pipeline with preprocessor to add
        :param input_data: data that are going to be passed through pipeline
        """
        structural_id = _get_pipeline_structural_id(pipeline, input_data)
        self._db.add_preprocessor(structural_id, pipeline.preprocessor)


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
    """
    Gets unique id from pipeline.

    :param pipeline: pipeline to get uid from
    :param input_data: data that are going to be passed through pipeline

    :return: unique pipeline and related data identificator
    """
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
