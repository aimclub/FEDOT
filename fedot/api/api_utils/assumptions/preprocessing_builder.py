from typing import Optional

from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.data.tensor_data import TensorData


class PreprocessingBuilder:
    """
    Builder for constructing preprocessing part of pipeline during the preparation of an initial assumption.
    """

    def __init__(self, task_type: TaskTypesEnum, data_type: DataTypesEnum, *initial_nodes: PipelineNode):
        self.task_type = task_type
        self.data_type = data_type
        self._builder = PipelineBuilder(*initial_nodes)

    @classmethod
    def builder(cls,
                task_type: TaskTypesEnum,
                data: TensorData,
                *initial_nodes: Optional[PipelineNode],
                use_optional_preprocessing: bool = True,
                ) -> PipelineBuilder:
        preprocessing_builder = cls(
            task_type,
            data.data_type,
            *initial_nodes,
        )
        if use_optional_preprocessing:
            return preprocessing_builder.with_optional_preprocessing()._builder
        return preprocessing_builder._builder

    def with_optional_preprocessing(self):
        """Add Tensor optional preprocessing node used by composer assumptions."""
        self._builder.add_node('optional_preprocessing')
        return self

    def to_pipeline(self, use_optional_preprocessing: bool = True) -> Optional[Pipeline]:
        """
        Returns result as Pipeline, optionally with optional preprocessing node.

        Returns:
            adapted graph as pipeline
        """
        if use_optional_preprocessing:
            return self.with_optional_preprocessing()._builder.build()
        return self._builder.build()
