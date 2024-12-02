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
        # All assumptions
        assumptions = {}

        # Constants
        CATBOOSTREG = 'catboostreg'
        XGBOOSTREG = 'xgboostreg'
        LGBMREG = 'lgbmreg'
        RFR = 'rfr'
        RIDGE = 'ridge'
        
        # Parameters of models
        models_params = {
            CATBOOSTREG: {
                "early_stopping_rounds": 30,
                "use_eval_set": True,
                "use_best_model": True
            },
            XGBOOSTREG: {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
            },
            LGBMREG: {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
            },
            RFR: {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            RIDGE: {}
        }

        # Get composite assumptions
        assumptions['gbm'] = PipelineBuilder() \
        .add_branch(CATBOOSTREG, XGBOOSTREG, LGBMREG) \
        .join_branches(CATBOOSTREG)

        # Get single-node assumptions
        single_models = [CATBOOSTREG, XGBOOSTREG, LGBMREG, RFR, RIDGE]

        for model in single_models:
            assumptions[model] = PipelineBuilder().add_node(model, params=models_params[model])

        return assumptions

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
        # All assumptions
        assumptions = {}

        # Constants
        CATBOOST = 'catboost'
        XGBOOST = 'xgboost'
        LGBM = 'lgbm'
        RF = 'rf'
        LOGIT = 'logit'
        
        # Parameters of models
        models_params = {
            CATBOOST: {
                "early_stopping_rounds": 30,
                "use_eval_set": True,
                "use_best_model": True
            },
            XGBOOST: {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
            },
            LGBM: {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
            },
            RF: {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            LOGIT: {}
        }

        # Get composite assumptions
        assumptions['gbm'] = PipelineBuilder() \
        .add_branch(CATBOOST, XGBOOST, LGBM) \
        .join_branches(CATBOOST)

        # Get single-node assumptions
        single_models = [CATBOOST, XGBOOST, LGBM, RF, LOGIT]

        for model in single_models:
            assumptions[model] = PipelineBuilder().add_node(model, params=models_params[model])

        return assumptions

    def ensemble_operation(self) -> str:
        return 'rf'

    def processing_builders(self) -> List[PipelineBuilder]:
        return list(self.builders.values())

    def fallback_builder(self, operations_filter: OperationsFilter) -> PipelineBuilder:
        random_choice_node = operations_filter.sample()
        return PipelineBuilder().add_node(random_choice_node)
