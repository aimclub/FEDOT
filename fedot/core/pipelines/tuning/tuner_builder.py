from datetime import timedelta
from typing import Callable, Type, Union

from hyperopt import tpe

from fedot.core.data.data import InputData
from fedot.core.optimisers.composer_requirements import ComposerRequirements
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.tuner_interface import HyperoptTuner
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import MetricType, MetricsEnum
from fedot.core.repository.tasks import Task
from fedot.utilities.define_metric_by_task import MetricByTask


class TunerBuilder:
    def __init__(self, task: Task):
        self.tuner_class = PipelineTuner
        self.cv_folds = None
        self.validation_blocks = None
        self.n_jobs = -1
        self.metric: MetricsEnum = MetricByTask.get_default_quality_metrics(task.task_type)[0]
        self.iterations = 100
        self.early_stopping_rounds = None
        self.timeout = timedelta(minutes=5)
        self.search_space = SearchSpace()
        self.algo = tpe.suggest
        self.eval_time_constraint = None

    def with_tuner(self, tuner: Type[HyperoptTuner]):
        self.tuner_class = tuner
        return self

    def with_requirements(self, requirements: ComposerRequirements):
        self.cv_folds = requirements.cv_folds
        self.validation_blocks = requirements.validation_blocks
        self.n_jobs = requirements.n_jobs
        return self

    def with_cv_folds(self, cv_folds: int):
        self.cv_folds = cv_folds
        return self

    def with_validation_blocks(self, validation_blocks: int):
        self.validation_blocks = validation_blocks
        return self

    def with_n_jobs(self, n_jobs: int):
        self.n_jobs = n_jobs
        return self

    def with_metric(self, metric: MetricType):
        self.metric = metric
        return self

    def with_iterations(self, iterations: int):
        self.iterations = iterations
        return self

    def with_early_stopping_rounds(self, early_stopping_rounds: int):
        self.early_stopping_rounds = early_stopping_rounds
        return self

    def with_timeout(self, timeout: timedelta):
        self.timeout = timeout
        return self

    def with_eval_time_constraint(self, eval_time_constraint: Union[timedelta, int, float]):
        self.eval_time_constraint = eval_time_constraint
        return self

    def with_search_space(self, search_space: SearchSpace):
        self.search_space = search_space
        return self

    def with_algo(self, algo: Callable):
        self.algo = algo
        return self

    def build(self, data: InputData) -> HyperoptTuner:
        objective = MetricsObjective(self.metric)
        data_producer = DataSourceSplitter(self.cv_folds, self.validation_blocks).build(data)
        objective_evaluate = PipelineObjectiveEvaluate(objective, data_producer,
                                                       validation_blocks=self.validation_blocks,
                                                       time_constraint=self.eval_time_constraint)
        tuner = self.tuner_class(objective_evaluate=objective_evaluate,
                                 iterations=self.iterations,
                                 early_stopping_rounds=self.early_stopping_rounds,
                                 timeout=self.timeout,
                                 search_space=self.search_space,
                                 algo=self.algo,
                                 n_jobs=self.n_jobs)
        return tuner
