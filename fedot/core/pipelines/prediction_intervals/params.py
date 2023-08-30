from typing import Union, Optional, List
from dataclasses import dataclass
from dataclasses import field

from golem.core.tuning.simultaneous import SimultaneousTuner


@dataclass
class PredictionIntervalsParams:
    """Class for defining the parameters for prediction intervals building process.

    Args:
        logging_level: logging level as in logging module
        n_jobs: n_jobs
        show_progress: show details of fitting process
        number_mutations: number mutations of final pipeline used in 'mutation_of_best_pipeline' method
        mutations_choice: a way to choose mutations. Possible options:

            - 'different' -> default value, all mutations are different
            - 'with_replacement' -> mutations taken randomly and can duplicate themselves

        mutations_discard_inapropriate_pipelines: delete/use strange pipelines in 'mutation_of_best_pipeline' method
        mutations_keep_percentage: percentage of pipelines to keep during 'mutation_of_best_pipeline' method;
                                  avaliable only if mutations_discard_inapropriate_pipelines=True
        mutations_operations: operations possible to generate mutations in 'mutation_of_best_pipeline' method;
        ql_number_models: number pipelines of the last generation in 'last_generation_ql' method. Avaliable option
                          'max' - use all possible pipelines
        ql_low_tuner, ql_up_tuner: tuner to tune hyperparameters of pipelines in 'last_generation_ql' method to
                                   make low/up prediction intervals. If None then default tuner is used.
        ql_tuner_iterations: number iterations of default tuner
        ql_tuner_minutes: number minutes for default tuner
        bpq_number_models: number pipelines of the last generation in 'best_pipelines_quantiles' method. Avaliable
                           option 'max' - use all possible pipelines.
    """

    logging_level: int = 20
    n_jobs: int = -1
    show_progress: bool = True
    number_mutations: int = 20
    mutations_choice: str = 'different'
    mutations_discard_inapropriate_pipelines: bool = True
    mutations_keep_percentage: float = 0.66
    mutations_operations: List[str] = field(default_factory=lambda: ['lagged', 'glm', 'ridge', 'sparse_lagged',
                                                                     'lasso', 'ts_naive_average', 'locf',
                                                                     'pca', 'linear', 'smoothing'])
    ql_number_models: Union[int, str] = 10
    ql_low_tuner: Optional[SimultaneousTuner] = None
    ql_up_tuner: Optional[SimultaneousTuner] = None
    ql_tuner_iterations: int = 10
    ql_tuner_minutes: int = 1
    bpq_number_models: Union[int, str] = 10
