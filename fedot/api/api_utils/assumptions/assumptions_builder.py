from typing import List, Union, Optional, Set

from fedot.api.api_utils.assumptions.operations_filter import OperationsFilter, WhitelistOperationsFilter
from fedot.api.api_utils.assumptions.preprocessing_builder import PreprocessingBuilder
from fedot.api.api_utils.assumptions.task_assumptions import TaskAssumptions
from fedot.core.log import default_log
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder, Node


class AssumptionsBuilder:

    def __init__(self, data: Union[InputData, MultiModalData], repository_name: str = 'model'):
        self.logger = default_log(prefix='FEDOT logger')
        self.data = data
        self.repo = OperationTypesRepository(repository_name)
        self.assumptions_generator = TaskAssumptions.for_task(self.data.task, self.repo)

    @staticmethod
    def get(data: Union[InputData, MultiModalData], repository_name: Optional[str] = None):
        if not repository_name:
            if data.data_type == DataTypesEnum.multi_ts:
                # It is needed to use data operations also for multi_ts data
                repository_name = 'all'
            else:
                repository_name = 'model'

        if isinstance(data, InputData):
            cls = UniModalAssumptionsBuilder
        elif isinstance(data, MultiModalData):
            cls = MultiModalAssumptionsBuilder
        else:
            raise NotImplementedError(f"Can't build assumptions for data type: {type(data).__name__}")
        return cls(data, repository_name=repository_name)

    def from_operations(self, available_operations: List[str]):
        raise NotImplementedError('abstract')

    def to_builders(self, initial_node: Optional[Node] = None) -> List[PipelineBuilder]:
        raise NotImplementedError('abstract')

    def build(self, initial_node: Optional[Node] = None) -> List[Pipeline]:
        return [builder.to_pipeline() for builder in self.to_builders(initial_node)]


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
            operations_to_choose_from = set(operations_for_task_and_data).intersection(available_operations)
            _check_operations_to_choose_from(self.data, self.data_type, operations_to_choose_from)
            if operations_to_choose_from:
                self.ops_filter = WhitelistOperationsFilter(available_operations, operations_to_choose_from)
            else:
                # Don't filter pipelines as we're not able to create
                # fallback pipelines without operations_to_choose_from.
                # So, leave default dumb ops_filter.
                self.logger.info(self.UNSUITABLE_AVAILABLE_OPERATIONS_MSG)
        return self

    def to_builders(self, initial_node: Optional[Node] = None) -> List[PipelineBuilder]:
        """ Return a list of valid builders satisfying internal
        OperationsFilter or a single fallback builder. """
        preprocessing = \
            PreprocessingBuilder.builder_for_data(self.data.task.task_type, self.data, initial_node)
        valid_builders = []
        for processing in self.assumptions_generator.processing_builders():
            candidate_builder = preprocessing.merge_with(processing)
            if self.ops_filter.satisfies(candidate_builder.to_pipeline()):
                valid_builders.append(candidate_builder)
        return valid_builders or [self.assumptions_generator.fallback_builder(self.ops_filter)]


class MultiModalAssumptionsBuilder(AssumptionsBuilder):
    def __init__(self, data: MultiModalData, repository_name: str = "model"):
        super().__init__(data, repository_name)
        _subbuilders = []
        for data_type, (data_source_name, values) in zip(self.data.data_type, self.data.items()):
            _subbuilders.append((data_source_name, UniModalAssumptionsBuilder(self.data, data_type)))
        self._subbuilders = tuple(_subbuilders)

    def from_operations(self, available_operations: Optional[List[str]] = None):
        for data_source, subbuilder in self._subbuilders:
            # Performs specific filter on image data operations
            if subbuilder.data_type is DataTypesEnum.image:
                available_img_operations = ['data_source_img', 'cnn']
                subbuilder.from_operations(available_img_operations)
        return self

    def to_builders(self, initial_node: Optional[Node] = None) -> List[PipelineBuilder]:
        # For each data source build its own list of alternatives of initial pipelines.
        subpipelines: List[List[Pipeline]] = []
        for data_source_name, subbuilder in self._subbuilders:
            first_node = PipelineBuilder().add_node(data_source_name).add_node(initial_node).to_nodes()[0]
            data_pipeline_alternatives = subbuilder.build(first_node)
            subpipelines.append(data_pipeline_alternatives)

        # Then zip these alternatives together and add final node to get ensembles.
        ensemble_builders: List[PipelineBuilder] = []
        for pre_ensemble in zip(*subpipelines):
            ensemble_operation = self.assumptions_generator.ensemble_operation()
            ensemble_nodes = map(lambda pipeline: pipeline.root_node, pre_ensemble)
            ensemble_builder = PipelineBuilder(*ensemble_nodes).join_branches(ensemble_operation)
            ensemble_builders.append(ensemble_builder)
        return ensemble_builders


def _check_operations_to_choose_from(data, data_type: DataTypesEnum, operations_to_choose_from: Set[str]):
    """Since it is sometimes impossible to form a valid pipeline without some operations,
     they are added to the set of operations for current task and data."""
    if isinstance(data, MultiModalData):
        if data_type is DataTypesEnum.image and 'data_source_img' not in operations_to_choose_from:
            operations_to_choose_from.add('data_source_img')
        if data_type is DataTypesEnum.text and 'data_source_text' not in operations_to_choose_from:
            operations_to_choose_from.add('data_source_text')
        if data_type is DataTypesEnum.table and 'data_source_table' not in operations_to_choose_from:
            operations_to_choose_from.add('data_source_table')
    if data_type is DataTypesEnum.image and 'cnn' not in operations_to_choose_from:
        operations_to_choose_from.add('cnn')
