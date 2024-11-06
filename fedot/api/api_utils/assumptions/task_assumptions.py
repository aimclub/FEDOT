from abc import abstractmethod
from typing import List

from fedot.api.api_utils.assumptions.operations_filter import OperationsFilter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class TaskAssumptions:
    """ Abstracts task-specific pipeline assumptions. """

    def __init__(self, repository: OperationTypesRepository):
        self.repo = repository

    @staticmethod
    def for_task(task: Task, repository: OperationTypesRepository) -> 'TaskAssumptions':
        assumptions_by_task = {
            TaskTypesEnum.classification: ClassificationAssumptions,
            TaskTypesEnum.regression: RegressionAssumptions,
            TaskTypesEnum.ts_forecasting: TSForecastingAssumptions,
        }
        assumptions_cls: TaskAssumptions = assumptions_by_task.get(task.task_type)
        if not assumptions_cls:
            raise NotImplementedError(f"Don't have assumptions for task type: {task.task_type}")
        return assumptions_cls(repository)

    @abstractmethod
    def ensemble_operation(self) -> str:
        """ Suitable ensemble operation used for MultiModalData case. """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def processing_builders(self) -> List[PipelineBuilder]:
        """ Returns alternatives of PipelineBuilders for core processing (without preprocessing). """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        """
        Returns default PipelineBuilder for case when primary alternatives are not valid.
        Have access for OperationsFilter for sampling available operations.
        """
        raise AbstractMethodNotImplementError


class TSForecastingAssumptions(TaskAssumptions):
    """ Simple static dictionary-based assumptions for time series forecasting task. """

    @property
    def builders(self):
        return {
            'lagged_ridge':
                PipelineBuilder()
            .add_sequence('lagged', 'ridge'),
            'topological':
                PipelineBuilder()
            .add_node('lagged')
            .add_node('topological_features')
            .add_node('lagged', branch_idx=1)
            .join_branches('ridge'),
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

    def processing_builders(self) -> List[PipelineBuilder]:
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
            # Composite assumptions
            'composite_gbm': self.build_composite_gbm(),
            'composite_linear_gbm': self.build_composite_linear_gbm(),

            # Single-node assumptions
            'catboostreg': PipelineBuilder().add_node('catboostreg'),
            'xgbreg': PipelineBuilder().add_node('xgbreg'),
            'lgbmreg': PipelineBuilder().add_node('lgbmreg'),
            'rfr': PipelineBuilder().add_node('rfr'),
            'ridge': PipelineBuilder().add_node('ridge')
        }
    
    def build_composite_gbm(self):
        return PipelineBuilder() \
            .add_branch('lgbmreg', 'catboostreg', 'xgbreg') \
            .join_branches('catboostreg')

    def build_composite_linear_gbm(self):
        return PipelineBuilder() \
            .add_node('ridge') \
            .add_branch('lgbmreg', 'catboostreg') \
            .join_branches('catboostreg')

    def ensemble_operation(self) -> str:
        return 'rfr'

    def processing_builders(self) -> List[PipelineBuilder]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        return PipelineBuilder().add_node(random_choice_node)


class ClassificationAssumptions(TaskAssumptions):
    """ Simple static dictionary-based assumptions for classification task. """

    @property
    def builders(self):
        # All pipelines
        pipelines = {}
        
        # Parameters of models
        models_params = {
            'catboost': {},
            'xgboost': {"early_stopping_rounds": 30},
            'lgbm': {"early_stopping_rounds": 30},
            'rf': {"n_jobs": 1},
            'logit': {}
        }

        # Get single-node pipelines
        single_models = ['catboost', 'xgboost', 'lgbm', 'rf', 'logit']

        for model in single_models:
            pipelines[model] = PipelineBuilder().add_node(model, params=models_params[model])

        # Get composite pipelines

        return pipelines

    def ensemble_operation(self) -> str:
        return 'rf'

    def processing_builders(self) -> List[PipelineBuilder]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        return PipelineBuilder().add_node(random_choice_node)
