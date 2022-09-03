from typing import Optional, Union

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_missing_values, data_has_categorical_features, \
    data_has_text_features
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


class PreprocessingBuilder:
    """
    Builder for constructing preprocessing part of pipeline during the preparation of an initial assumption.
    If data is multimodal, builder makes preprocessing pipeline for each data source iteratively.
    """
    def __init__(self, task_type: TaskTypesEnum, data_type: DataTypesEnum, *initial_nodes: Node):
        self.task_type = task_type
        self.data_type = data_type
        self._builder = PipelineBuilder(*initial_nodes)

    @classmethod
    def builder_for_data(cls,
                         task_type: TaskTypesEnum,
                         data: Union[InputData, MultiModalData],
                         *initial_nodes: Optional[Node]) -> PipelineBuilder:
        if isinstance(data, MultiModalData):
            # if the data is unimodal, initial_nodes = tuple of None
            # if the data is multimodal, initial_nodes = tuple of 1 element (current data_source node)
            # so the whole data is reduced to the current data_source for an easier preprocessing
            data = data[str(initial_nodes[0])]
        preprocessing_builder = cls(task_type, data.data_type, *initial_nodes)
        if data_has_text_features(data):
            preprocessing_builder = preprocessing_builder.with_text_vectorizer()
        return preprocessing_builder.to_builder()

    def with_scaling(self):
        if self.task_type is not TaskTypesEnum.ts_forecasting and self.data_type is not DataTypesEnum.image:
            self._builder.add_node('scaling')
        return self

    def with_text_vectorizer(self):
        self._builder.add_node('tfidf')
        return self

    def to_builder(self) -> PipelineBuilder:
        """ Return result as PipelineBuilder. Scaling is applied final by default. """
        return self.with_scaling()._builder

    def to_pipeline(self) -> Optional[Pipeline]:
        """ Return result as Pipeline. Scaling is applied final by default. """
        return self.to_builder().to_pipeline()
