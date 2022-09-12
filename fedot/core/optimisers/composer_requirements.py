import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComposerRequirements:
    """Defines general algorithm-independent parameters of the composition process
    (like stop condition, validation, timeout, logging etc.)

    Options related to stop condition:
    :param num_of_generations: maximum number of optimizer generations
    :param timeout: max time in minutes available for composition process
    :param early_stopping_generations: optional max number of stagnating
    populations for early stopping. If None -- do not use early stopping.

    Infrastructure options (logging, performance)
    :param keep_n_best: number of the best individuals of previous generation to keep in next generation
    :param max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :param n_jobs: num of n_jobs
    :param show_progress: bool indicating whether to show progress using tqdm or not
    :param collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline

    Model validation options:
    :param cv_folds: number of cross-validation folds
    :param validation_blocks: number of validation blocks for time series validation
    """

    num_of_generations: int = 20
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    early_stopping_generations: Optional[int] = 10

    keep_n_best: int = 1
    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    n_jobs: int = 1
    show_progress: bool = True
    collect_intermediate_metric: bool = False

    cv_folds: Optional[int] = None
    validation_blocks: Optional[int] = None

    def __post_init__(self):
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError('Number of folds for KFold cross validation must be 2 or more.')
