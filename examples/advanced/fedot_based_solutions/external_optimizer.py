from typing import Any, Optional, Sequence, Union

from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.dag.graph import Graph
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.objective import Objective, ObjectiveFunction
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.utils import fedot_project_root


class RandomMutationSearchOptimizer(GraphOptimiser):
    """
    Random search-based graph models optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Graph, Sequence[Graph]] = (),
                 requirements: Optional[Any] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 parameters: Optional[GraphOptimiserParameters] = None,
                 log: Optional[Log] = None):
        super().__init__(objective, initial_graph, requirements, graph_generation_params, parameters, log)
        self.change_types = [boosting_mutation, parameter_change_mutation,
                             MutationTypesEnum.single_edge,
                             MutationTypesEnum.single_change,
                             MutationTypesEnum.single_drop,
                             MutationTypesEnum.single_add]

    def optimise(self, objective: ObjectiveFunction, show_progress: bool = True):

        timer = OptimisationTimer(log=self.log, timeout=self.requirements.timeout)
        dispatcher = MultiprocessingDispatcher(self.graph_generation_params.adapter, timer, log=self.log, n_jobs=1)
        evaluator = dispatcher.dispatch(objective)

        num_iter = 0
        best = Individual(self.initial_graphs)
        evaluator([best])

        with timer as t:
            while not t.is_time_limit_reached(num_iter):
                new = mutation(types=self.change_types, ind=best, params=self.graph_generation_params,
                               requirements=self.requirements, log=self.log)
                evaluator([new])
                if new.fitness < best.fitness:
                    best = new
                num_iter += 1

        return [best.graph]


def run_with_random_search_composer():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    composer_params = {'available_operations': ['class_decompose', 'rf', 'linear', 'xgboost', 'dt'],
                       'optimizer': RandomMutationSearchOptimizer}

    automl = Fedot(problem='classification', timeout=1, verbose_level=4,
                   preset='fast_train', composer_params=composer_params)

    automl.fit(train_data_path)
    automl.predict(test_data_path)
    print(automl.get_metrics())


if __name__ == '__main__':
    run_with_random_search_composer()
