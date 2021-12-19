from typing import Any, Callable, List, Optional, Union

from fedot.api.main import Fedot
from fedot.core.log import Log
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.repository.quality_metrics_repository import (MetricsEnum)
from fedot.core.utils import fedot_project_root


class RandomSearchOptimizer(GraphOptimiser):
    def __init__(self, initial_graph: Union[Any, List[Any]],
                 requirements: Any,
                 graph_generation_params: GraphGenerationParams,
                 metrics: List[MetricsEnum],
                 parameters: GraphOptimiserParameters = None,
                 log: Log = None):
        super().__init__(initial_graph, requirements, graph_generation_params, metrics, parameters, log)

    def optimise(self, objective_function,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True):
        return self.initial_graph


def run_with_random_search_composer():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    automl = Fedot(problem='classification', timeout=1, verbose_level=4,
                   preset='light_notun')
    automl.api_composer.optimiser = RandomSearchOptimizer

    automl.fit(train_data_path)
    automl.predict(test_data_path)


run_with_random_search_composer()
