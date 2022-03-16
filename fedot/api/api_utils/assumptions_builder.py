from random import choice
from typing import List, Union, Optional

from fedot.core.log import Log, default_log
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder, Node


class OperationsFilter:
    def satisfies(self, pipeline: Pipeline) -> bool:
        """ Checks if all operations in a Pipeline satisify this filter. """
        return True

    def sample(self) -> str:
        """ Samples some operation that satisfies this filter. """
        raise NotImplementedError()


class WhitelistOperationsFilter(OperationsFilter):
    """ Simple OperationsFilter implementation based on two lists:
    one for all available operations, another for sampling operations. """

    def __init__(self, available_ops: List[str], available_task_ops: Optional[List[str]] = None):
        super().__init__()
        self._whitelist = tuple(available_ops)
        self._choice_ops = tuple(available_task_ops) if available_task_ops else self._whitelist

    def satisfies(self, pipeline: Pipeline) -> bool:
        def node_ok(node):
            return node.operation.operation_type in self._whitelist

        return all(map(node_ok, pipeline.nodes))

    def sample(self) -> str:
        return choice(self._choice_ops)


class TaskAssumptions:
    """ Abstracts task-specific pipeline assumptions from preprocessing and conditioned pipeline building. """

    @staticmethod
    def for_task(task: Task):
        task_to_assumptions = {
            TaskTypesEnum.classification: ClassificationAssumptions,
            TaskTypesEnum.regression: RegressionAssumptions,
            TaskTypesEnum.ts_forecasting: TSForecastingAssumptions,
        }
        assumptions_cls = task_to_assumptions.get(task.task_type)
        if not assumptions_cls:
            raise NotImplementedError(f"Don't have assumptions for task type: {task.task_type}")
        return assumptions_cls()

    def ensemble_operation(self) -> str:
        """ Suitable ensemble operation used for MultiModalData case. """
        raise NotImplementedError()

    def processing_pipelines(self, node_preprocessed: Optional[Node]) -> List[Pipeline]:
        """ Returns alternatives of core Pipelines (without preprocessing). """
        raise NotImplementedError()

    def fallback_pipeline(self, ops_filter: OperationsFilter, initial_node: Optional[Node] = None) -> Pipeline:
        """
        Returns default Pipeline in the case when primary alternatives
        from .processing_pipelines() didn't pass OperationsFilter.
        Have access for OperationsFilter for sampling available operations.
        """
        raise NotImplementedError()


class TSForecastingAssumptions(TaskAssumptions):

    def ensemble_operation(self) -> str:
        return 'ridge'

    def processing_pipelines(self, node_preprocessed: Optional[Node] = None) -> List[Pipeline]:
        return [
            self.create_glm_ridge_pipeline(node_preprocessed),
            self.create_lagged_ridge_pipeline(node_preprocessed),
            self.create_polyfit_ridge_pipeline(node_preprocessed),
            self.create_ar_pipeline(node_preprocessed)
        ]

    def fallback_pipeline(self, ops_filter: OperationsFilter, initial_node: Optional[Node] = None) -> Pipeline:
        random_choice_node = ops_filter.sample()
        return PipelineBuilder(initial_node).add_node('lagged').add_node(random_choice_node).to_pipeline()

    @staticmethod
    def create_lagged_ridge_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed) \
            .add_node('lagged') \
            .add_node('ridge') \
            .to_pipeline()

    @staticmethod
    def create_glm_ridge_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed) \
            .add_branch('glm', 'lagged') \
            .add_node('ridge', branch_idx=1) \
            .join_branches('ridge') \
            .to_pipeline()

    @staticmethod
    def create_polyfit_ridge_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed) \
            .add_branch('polyfit', 'lagged') \
            .grow_branches(None, 'ridge') \
            .join_branches('ridge') \
            .to_pipeline()

    @staticmethod
    def create_ar_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed).add_sequence('smoothing', 'ar').to_pipeline()


class RegressionAssumptions(TaskAssumptions):

    def ensemble_operation(self) -> str:
        return 'rfr'

    def processing_pipelines(self, node_preprocessed: Optional[Node] = None) -> List[Pipeline]:
        return [self.create_rfr_regression_pipeline(node_preprocessed),
                self.create_ridge_regression_pipeline(node_preprocessed)]

    def fallback_pipeline(self, ops_filter: OperationsFilter, initial_node: Optional[Node] = None) -> Pipeline:
        random_choice_node = ops_filter.sample()
        return PipelineBuilder(initial_node).add_node(random_choice_node).to_pipeline()

    @staticmethod
    def create_rfr_regression_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed).add_node('rfr').to_pipeline()

    @staticmethod
    def create_ridge_regression_pipeline(node_preprocessed):
        return PipelineBuilder(node_preprocessed).add_node('ridge').to_pipeline()


class ClassificationAssumptions(TaskAssumptions):

    def ensemble_operation(self) -> str:
        return 'rf'

    def processing_pipelines(self, node_preprocessed: Optional[Node] = None) -> List[Pipeline]:
        return [self.create_rf_classifier_pipeline(node_preprocessed),
                self.create_logit_classifier_pipeline(node_preprocessed)]

    def fallback_pipeline(self, ops_filter: OperationsFilter, initial_node: Optional[Node] = None) -> Pipeline:
        random_choice_node = ops_filter.sample()
        return PipelineBuilder(initial_node).add_node(random_choice_node).to_pipeline()

    @staticmethod
    def create_rf_classifier_pipeline(node_preprocessed: Optional[Node]):
        return PipelineBuilder(node_preprocessed).add_node('rf').to_pipeline()

    @staticmethod
    def create_logit_classifier_pipeline(node_preprocessed):
        return PipelineBuilder(node_preprocessed).add_node('logit').to_pipeline()


class PreprocessingBuilder:
    @classmethod
    def build_for_data(cls,
                       task_type: TaskTypesEnum,
                       data: Union[InputData, MultiModalData],
                       *initial_nodes: Optional[Node]) -> Optional[Node]:
        preprocessing_builder = cls(task_type, *initial_nodes)
        if data_has_missing_values(data):
            preprocessing_builder = preprocessing_builder.with_gaps()
        if data_has_categorical_features(data):
            preprocessing_builder = preprocessing_builder.with_categorical()
        return next(iter(preprocessing_builder.to_nodes()), None)  # first or none

    def __init__(self, task_type: TaskTypesEnum, *initial_nodes: Node):
        self.task_type = task_type
        self._builder = PipelineBuilder(*initial_nodes)

    def with_gaps(self):
        self._builder.add_node('simple_imputation')
        return self

    def with_categorical(self):
        if self.task_type != TaskTypesEnum.ts_forecasting:
            self._builder.add_node('one_hot_encoding')
        return self

    def with_scaling(self) -> PipelineBuilder:
        if self.task_type != TaskTypesEnum.ts_forecasting:
            self._builder.add_node('scaling')
        return self._builder

    def to_nodes(self) -> List[Node]:
        """ Return result as list of nodes. Scaling is applied final by default. """
        return self.with_scaling().to_nodes()

    def to_pipeline(self) -> Optional[Pipeline]:
        """ Return result as Pipeline. Scaling is applied final by default. """
        return self.with_scaling().to_pipeline()


class AssumptionsBuilder:

    def __init__(self, task: Task, data: Union[InputData, MultiModalData]):
        self.logger = default_log('FEDOT logger')
        self.data = data
        self.task = task
        self.assumptions_generator = TaskAssumptions.for_task(task)

    @staticmethod
    def get(task: Task, data: Union[InputData, MultiModalData]):
        if isinstance(data, InputData):
            return UnimodalAssumptionsBuilder(task, data)
        elif isinstance(data, MultiModalData):
            return MultiModalAssumptionsBuilder(task, data)
        else:
            raise NotImplementedError(f"Can't build assumptions for data type: {type(data).__name__}")

    def with_logger(self, logger: Log):
        raise NotImplementedError('abstract')

    def from_operations(self, available_ops: List[str]):
        raise NotImplementedError('abstract')

    def build(self, initial_node: Optional[Node] = None) -> List[Pipeline]:
        raise NotImplementedError('abstract')


class UnimodalAssumptionsBuilder(AssumptionsBuilder):
    UNSUITABLE_AVAILABLE_OPERATIONS_MSG = "Unable to construct an initial assumption from the passed " \
                                          "available operations, default initial assumption will be used"

    def __init__(self, task: Task, data: Union[InputData, MultiModalData], data_type: DataTypesEnum = None):
        """ Construct builder from task and data.
        :param task: task for the pipeline
        :param data: data that will be passed to the pipeline
        :param data_type: allows specifying data_type of particular column for MultiModalData case
        """
        super().__init__(task, data)
        self.data_type = data_type or data.data_type
        self.ops_filter = OperationsFilter()

    def with_logger(self, logger: Log):
        self.logger = logger
        return self

    def from_operations(self, available_ops: Optional[List[str]]):
        if available_ops:
            operations_to_choose_from = \
                self._get_operations_for_the_task(task_type=self.task.task_type, data_type=self.data_type,
                                                  repo='model', available_operations=available_ops)
            if operations_to_choose_from:
                self.ops_filter = WhitelistOperationsFilter(available_ops, operations_to_choose_from)
            else:
                # Don't filter pipelines as we're not able to create
                # fallback pipelines without operations_to_choose_from.
                # So, leave default dumb ops_filter.
                self.logger.message(self.UNSUITABLE_AVAILABLE_OPERATIONS_MSG)
        return self

    def build(self, initial_node: Optional[Node] = None) -> List[Pipeline]:
        def _filter_or_fallback(pipeline: Pipeline) -> Pipeline:
            if self.ops_filter.satisfies(pipeline):
                return pipeline
            else:
                return self.assumptions_generator.fallback_pipeline(self.ops_filter, initial_node)

        node_preprocessed: Optional[Node] = \
            PreprocessingBuilder.build_for_data(self.task.task_type, self.data, initial_node)
        candidate_pipelines = self.assumptions_generator.processing_pipelines(node_preprocessed)
        valid_pipelines = list(map(_filter_or_fallback, candidate_pipelines))

        return valid_pipelines

    @staticmethod
    def _get_operations_for_the_task(task_type: TaskTypesEnum, data_type: DataTypesEnum, repo: str,
                                     available_operations: List[str]):
        """ Returns the intersection of the sets of passed available operations and
        operations that are suitable for solving the given problem """
        operations_for_the_task, _ = \
            OperationTypesRepository(repo).suitable_operation(task_type=task_type, data_type=data_type)
        operations_to_choose_from = list(set(operations_for_the_task).intersection(available_operations))
        return operations_to_choose_from


class MultiModalAssumptionsBuilder(AssumptionsBuilder):
    def __init__(self, task: Task, data: MultiModalData):
        super().__init__(task, data)
        _subbuilders = []
        for data_type, (data_source_name, values) in zip(data.data_type, data.items()):
            # TODO: can have specific Builder for each particular data column, eg construct InputData
            _subbuilders.append((data_source_name, UnimodalAssumptionsBuilder(task, data, data_type=data_type)))
        self._subbuilders = tuple(_subbuilders)

    def with_logger(self, logger: Log):
        self.logger = logger
        for _, subbuilder in self._subbuilders:
            subbuilder.with_logger(logger)
        return self

    # TODO: in principle, each data column in MultiModalData can have its own available_ops
    def from_operations(self, available_ops: List[str]):
        self.logger.message("Available operations are not taken into account when "
                            "forming the initial assumption for multi-modal data")
        # for _, subbuilder in self._subbuilders:
        #     subbuilder.from_operations(available_ops)
        return self

    def build(self, initial_node: Optional[Node] = None) -> List[Pipeline]:
        # For each data source build its own list of alternatives of initial pipelines.
        subpipelines: List[List[Pipeline]] = []
        for data_source_name, subbuilder in self._subbuilders:
            first_node = PipelineBuilder().add_node(data_source_name).add_node(initial_node).to_nodes()[0]
            data_pipeline_alternatives = subbuilder.build(first_node)
            subpipelines.append(data_pipeline_alternatives)

        # Then zip these alternatives together and add final node to get ensembles.
        ensembles: List[Pipeline] = []
        for pre_ensemble in zip(*subpipelines):
            node_final = self.assumptions_generator.ensemble_operation()
            ensemble_nodes = map(lambda pipeline: pipeline.root_node, pre_ensemble)
            ensemble_pipeline = PipelineBuilder(*ensemble_nodes).join_branches(node_final).to_pipeline()
            ensembles.append(ensemble_pipeline)
        return ensembles
