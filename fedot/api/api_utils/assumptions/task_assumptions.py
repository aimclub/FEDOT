from typing import List

from fedot.api.api_utils.assumptions.operations_filter import OperationsFilter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum


class TaskAssumptions:
    """ Abstracts task-specific pipeline assumptions. """

    def __init__(self, repository: OperationTypesRepository):
        self.repo = repository

    @staticmethod
    def for_task(task: Task, repository: OperationTypesRepository):
        assumptions_by_task = {
            TaskTypesEnum.classification: ClassificationAssumptions,
            TaskTypesEnum.regression: RegressionAssumptions,
            TaskTypesEnum.ts_forecasting: TSForecastingAssumptions,
        }
        assumptions_cls = assumptions_by_task.get(task.task_type)
        if not assumptions_cls:
            raise NotImplementedError(f"Don't have assumptions for task type: {task.task_type}")
        return assumptions_cls(repository)

    def ensemble_operation(self) -> str:
        """ Suitable ensemble operation used for MultiModalData case. """
        raise NotImplementedError()

    def processing_builders(self) -> List[PipelineBuilder]:
        """ Returns alternatives of PipelineBuilders for core processing (without preprocessing). """
        raise NotImplementedError()

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        """
        Returns default PipelineBuilder for case when primary alternatives are not valid.
        Have access for OperationsFilter for sampling available operations.
        """
        raise NotImplementedError()


class TSForecastingAssumptions(TaskAssumptions):
    """ Simple static dictionary-based assumptions for time series forecasting task. """

    @property
    def builders(self):
        return {
            'glm_ridge':
                PipelineBuilder()
                .add_branch('glm', 'lagged')
                .add_node('ridge', branch_idx=1)
                .join_branches('ridge'),
            'lagged_ridge':
                PipelineBuilder()
                .add_sequence('lagged', 'ridge'),
            'polyfit_ridge':
                PipelineBuilder()
                .add_branch('polyfit', 'lagged')
                .grow_branches(None, 'ridge')
                .join_branches('ridge'),
            'smoothing_ar':
                PipelineBuilder()
                .add_sequence('smoothing', 'ar'),
        }

    def ensemble_operation(self) -> str:
        return 'ridge'

    def processing_builders(self) -> List[Pipeline]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        operation_info = self.repo.operation_info_by_id(random_choice_node)
        if 'non_lagged' in operation_info.tags:
            return PipelineBuilder().add_node(random_choice_node)
        else:
            return PipelineBuilder().add_node('lagged').add_node(random_choice_node)


class RegressionAssumptions(TaskAssumptions):
    """ Simple static dictionary-based assumptions for regression task. """

    @property
    def builders(self):
        return {
            'rfr': PipelineBuilder().add_node('rfr'),
            'ridge': PipelineBuilder().add_node('ridge'),
        }

    def ensemble_operation(self) -> str:
        return 'rfr'

    def processing_builders(self) -> List[Pipeline]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        return PipelineBuilder().add_node(random_choice_node)


class ClassificationAssumptions(TaskAssumptions):
    """ Simple static dictionary-based assumptions for classification task. """

    @property
    def builders(self):
        return {
            'rf': PipelineBuilder().add_node('rf'),
            'logit': PipelineBuilder().add_node('logit'),
            'cnn': PipelineBuilder().add_node('cnn'),
        }

    def ensemble_operation(self) -> str:
        return 'rf'

    def processing_builders(self) -> List[Pipeline]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        return PipelineBuilder().add_node(random_choice_node)
