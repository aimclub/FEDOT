from typing import Union, Optional

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import data_has_missing_values, data_has_categorical_features
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import TaskTypesEnum


class PreprocessingBuilder:

    def __init__(self, task_type: TaskTypesEnum, *initial_nodes: Node):
        self.task_type = task_type
        self._builder = PipelineBuilder(*initial_nodes)

    @classmethod
    def builder_for_data(cls,
                         task_type: TaskTypesEnum,
                         data: Union[InputData, MultiModalData],
                         *initial_nodes: Optional[Node]) -> PipelineBuilder:
        preprocessing_builder = cls(task_type, *initial_nodes)
        if data_has_missing_values(data):
            preprocessing_builder = preprocessing_builder.with_gaps()
        if data_has_categorical_features(data):
            preprocessing_builder = preprocessing_builder.with_categorical()
        return preprocessing_builder.to_builder()

    def with_gaps(self):
        self._builder.add_node('simple_imputation')
        return self

    def with_categorical(self):
        if self.task_type != TaskTypesEnum.ts_forecasting:
            self._builder.add_node('one_hot_encoding')
        return self

    def with_scaling(self):
        if self.task_type != TaskTypesEnum.ts_forecasting:
            self._builder.add_node('scaling')
        return self

    def to_builder(self) -> PipelineBuilder:
        """ Return result as PipelineBuilder. Scaling is applied final by default. """
        return self.with_scaling()._builder

    def to_pipeline(self) -> Optional[Pipeline]:
        """ Return result as Pipeline. Scaling is applied final by default. """
        return self.to_builder().to_pipeline()
