import datetime
from dataclasses import dataclass, field
from typing import Optional

from fedot.core.utils import default_fedot_data_dir


@dataclass
class ComposerRequirements:
    """Defines general algorithm-independent parameters of the composition process
    (like stop condition, validation, timeout, logging etc.)

    Options related to stop condition:
    :param num_of_generations: maximum number of optimizer generations
    :param timeout: max time in minutes available for composition process
    :param early_stopping_iterations: optional max number of stagnating
    populations for early stopping. If None -- do not use early stopping.

    Infrastructure options (logging, performance)
    :param keep_n_best: number of the best individuals of previous generation to keep in next generation
    :param max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :param n_jobs: num of n_jobs
    :param show_progress: bool indicating whether to show progress using tqdm or not
    :param collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline

    History options:
    :param keep_history: if True, then save generations to history; if False, don't keep history.
    :param history_dir: directory for saving optimization history, optional.
      If the path is relative, then save relative to `default_fedot_data_dir`.
      If absolute -- then save directly by specified path.
      If None -- do not save the history to disk and keep it only in-memory.

    Model validation options:
    :param cv_folds: number of cross-validation folds
    :param validation_blocks: number of validation blocks for time series validation
    """

    num_of_generations: Optional[int] = None
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    early_stopping_iterations: Optional[int] = 10
    early_stopping_timeout: Optional[float] = 5
    keep_n_best: int = 1
    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    n_jobs: int = 1
    show_progress: bool = True
    collect_intermediate_metric: bool = False

    keep_history: bool = True
    history_dir: Optional[str] = field(default_factory=default_fedot_data_dir)

    cv_folds: Optional[int] = None
    validation_blocks: Optional[int] = None

    def __post_init__(self):
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError('Number of folds for KFold cross validation must be 2 or more.')
