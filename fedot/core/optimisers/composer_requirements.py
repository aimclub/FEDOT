import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ComposerRequirements:
    """Defines infrastructural and algorithm options for optimization process.

    :param primary: operation types for :class:`~fedot.core.pipelines.node.PrimaryNode`s
    :param secondary: operation types for :class:`~fedot.core.pipelines.node.SecondaryNode`s

    Options related to stop condition:
    :param num_of_generations: maximal number of evolutionary algorithm generations
    :param timeout: max time in minutes available for composition process
    :param early_stopping_generations: optional max number of stagnating populations for early stopping.
    Do not use early stopping if None.

    Populational optimization options
    :param offspring_rate: offspring rate used on next population
    :param pop_size: initial population size; if unspecified, default value is used
    :param max_pop_size: maximum population size; optional, if unspecified, then population size is unbound
    :param keep_n_best: number of the best individuals of previous generation to keep in next generation
    :param adaptive_depth: flag to enable adaptive configuration of graph depth
    :param adaptive_depth_max_stagnation: max number of stagnating populations before adaptive depth increment

    Infrastructure options (logging, performance)
    :param max_pipeline_fit_time: time constraint for operation fitting (minutes)
    :param n_jobs: num of n_jobs
    :param show_progress: bool indicating whether to show progress using tqdm or not
    :param logging_level_opt: level of logging in optimizer
    :param collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline

    Model validation options:
    :param cv_folds: number of cross-validation folds
    :param validation_blocks: number of validation blocks for time series validation
    """
    primary: Sequence[str] = tuple()
    secondary: Sequence[str] = tuple()

    num_of_generations: int = 20
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    early_stopping_generations: Optional[int] = 10

    offspring_rate: float = 0.5
    pop_size: int = 20
    max_pop_size: Optional[int] = 55
    keep_n_best: int = 1
    adaptive_depth: bool = False
    adaptive_depth_max_stagnation: int = 3

    max_pipeline_fit_time: Optional[datetime.timedelta] = None
    n_jobs: int = 1
    show_progress: bool = True
    logging_level_opt: int = logging.INFO
    collect_intermediate_metric: bool = False

    cv_folds: Optional[int] = None
    validation_blocks: Optional[int] = None

    def __post_init__(self):
        if self.cv_folds is not None and self.cv_folds <= 1:
            raise ValueError(f'Number of folds for KFold cross validation must be 2 or more.')
