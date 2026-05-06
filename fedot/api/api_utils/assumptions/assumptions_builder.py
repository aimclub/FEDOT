from abc import abstractmethod
from typing import List, Union, Optional, Tuple

from golem.core.log import default_log

from fedot.api.api_utils.assumptions.assumption_rules import (
    build_operations_filter_decision,
    default_repository_name_for_data,
    normalize_assumption_data_type,
)
from fedot.api.api_utils.assumptions.operations_filter import OperationsFilter, WhitelistOperationsFilter
from fedot.api.api_utils.assumptions.preprocessing_builder import PreprocessingBuilder
from fedot.api.api_utils.assumptions.task_assumptions import TaskAssumptions
from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class AssumptionsBuilder:

    def __init__(self, data: Union[InputData, MultiModalData], repository_name: str = 'model'):
        self.logger = default_log(prefix='FEDOT logger')
        self.data = data
        self.repo = OperationTypesRepository(repository_name)
        self.assumptions_generator = TaskAssumptions.for_task(self.data.task, self.repo)

    @staticmethod
    def get(data: Union[InputData, MultiModalData], repository_name: Optional[str] = None):
        if not repository_name:
            repository_name = default_repository_name_for_data(data)

        if isinstance(data, InputData):
            cls = UniModalAssumptionsBuilder
        elif isinstance(data, MultiModalData):
            cls = MultiModalAssumptionsBuilder
        else:
            raise NotImplementedError(f"Can't build assumptions for data type: {type(data).__name__}")
        return cls(data, repository_name=repository_name)

    @abstractmethod
    def from_operations(self, available_operations: List[str]):
        raise AbstractMethodNotImplementError

    @abstractmethod
    def to_builders(self, initial_node: Optional[PipelineNode] = None,
                    use_input_preprocessing: bool = True) -> List[PipelineBuilder]:
        raise AbstractMethodNotImplementError

    def build(self, initial_node: Optional[PipelineNode] = None,
              use_input_preprocessing: bool = True) -> List[Pipeline]:
        return [
            builder.build()
            for builder in self.to_builders(initial_node, use_input_preprocessing=use_input_preprocessing)]


class UniModalAssumptionsBuilder(AssumptionsBuilder):
    UNSUITABLE_AVAILABLE_OPERATIONS_MSG = "Unable to construct an initial assumption from the passed " \
                                          "available operations, default initial assumption will be used"

    def __init__(self, data: Union[InputData, MultiModalData],
                 data_type: DataTypesEnum = None, repository_name: str = "model"):
        """ Construct builder from task and data.
        :param data: data that will be passed to the pipeline
        :param data_type: allows specifying data_type of particular column for MultiModalData case
        """
        super().__init__(data, repository_name)
        self.data_type = data_type or data.data_type
        self.ops_filter = OperationsFilter()

    def from_operations(self, available_operations: Optional[List[str]] = None):
        if available_operations:
            operations_for_task_and_data = self.repo.suitable_operation(self.data.task.task_type, self.data_type)
            filter_decision = build_operations_filter_decision(
                data=self.data,
                data_type=self.data_type,
                available_operations=available_operations,
                suitable_operations=operations_for_task_and_data,
            )
            if filter_decision.allow_filtering:
                self.ops_filter = WhitelistOperationsFilter(
                    filter_decision.whitelist,
                    filter_decision.sampling_choices,
                )
            else:
                # Don't filter pipelines as we're not able to create
                # fallback pipelines without sampling choices.
                # So, leave default dumb ops_filter.
                self.logger.info(self.UNSUITABLE_AVAILABLE_OPERATIONS_MSG)
        return self

    def to_builders(self, initial_node: Optional[PipelineNode] = None,
                    use_input_preprocessing: bool = True) -> List[PipelineBuilder]:
        """ Return a list of valid builders satisfying internal
        OperationsFilter or a single fallback builder. """
        preprocessing = \
            PreprocessingBuilder.builder_for_data(self.data.task.task_type, self.data, initial_node,
                                                  use_input_preprocessing=use_input_preprocessing)
        valid_builders = []
        for processing in self.assumptions_generator.processing_builders():
            candidate_builder = preprocessing.merge_with(processing)
            if self.ops_filter.satisfies(candidate_builder.build()):
                valid_builders.append(candidate_builder)
        return valid_builders or [self.assumptions_generator.fallback_builder(self.ops_filter)]


class MultiModalAssumptionsBuilder(AssumptionsBuilder):
    def __init__(self, data: MultiModalData, repository_name: str = "model"):
        super().__init__(data, repository_name)
        _subbuilders = []
        for data_type, (data_source_name, _) in zip(self.data.data_type, self.data.items()):
            _subbuilders.append((data_source_name, UniModalAssumptionsBuilder(self.data, data_type)))
        self._subbuilders: Tuple[Tuple[str, UniModalAssumptionsBuilder]] = tuple(_subbuilders)

    def from_operations(self, available_operations: Optional[List[str]] = None):
        for _, subbuilder in self._subbuilders:
            # TS tensor modalities (incl. legacy ``image``) use time-series data source + conv models
            if normalize_assumption_data_type(subbuilder.data_type) is DataTypesEnum.ts:
                available_ts_tensor_operations = ['data_source_time_series', 'cnn']
                subbuilder.from_operations(available_ts_tensor_operations)
        return self

    def to_builders(self, initial_node: Optional[PipelineNode] = None,
                    use_input_preprocessing: bool = True) -> List[PipelineBuilder]:
        # For each data source build its own list of alternatives of initial pipelines.
        subpipelines: List[List[Pipeline]] = []
        initial_node_operation = initial_node.operation.operation_type if initial_node is not None else None
        for data_source_name, subbuilder in self._subbuilders:
            first_node = PipelineBuilder(use_input_preprocessing=use_input_preprocessing) \
                .add_node(data_source_name).add_node(initial_node_operation).to_nodes()[0]
            data_pipeline_alternatives = subbuilder.build(first_node, use_input_preprocessing=use_input_preprocessing)
            subpipelines.append(data_pipeline_alternatives)

        # TODO: fix this workaround during the improvement of multi-modality
        for i, subpipeline in enumerate(subpipelines):
            if (len(subpipeline) == 1 and len(subpipeline[0].nodes) == 1 and
                    str(subpipeline[0].nodes[0]) in ['cnn', 'data_source_time_series', 'data_source_img']):
                subpipelines[i] = [Pipeline(PipelineNode('cnn', nodes_from=[PipelineNode('data_source_time_series')]))]

        # Then zip these alternatives together and add final node to get ensembles.
        ensemble_builders: List[PipelineBuilder] = []
        for pre_ensemble in zip(*subpipelines):
            ensemble_operation = self.assumptions_generator.ensemble_operation()
            ensemble_nodes = map(lambda pipeline: pipeline.root_node, pre_ensemble)
            ensemble_builder = PipelineBuilder(*ensemble_nodes, use_input_preprocessing=use_input_preprocessing) \
                .join_branches(ensemble_operation)
            ensemble_builders.append(ensemble_builder)
        return ensemble_builders
