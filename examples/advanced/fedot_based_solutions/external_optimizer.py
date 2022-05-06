from typing import Any, Callable, List, Optional, Union

from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.evaluation import EvaluationDispatcher
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.utils import fedot_project_root
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate


class RandomSearchOptimizer(GraphOptimiser):
    """
    Random search-based graph models optimizer
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 objective: Objective,
                 requirements: Any,
                 graph_generation_params: GraphGenerationParams,
                 parameters: GraphOptimiserParameters = None,
                 log: Optional[Log] = None):
        super().__init__(initial_graph, objective, requirements, graph_generation_params, parameters, log)
        self.change_types = [boosting_mutation, parameter_change_mutation,
                             MutationTypesEnum.single_edge,
                             MutationTypesEnum.single_change,
                             MutationTypesEnum.single_drop,
                             MutationTypesEnum.single_add]

    def optimise(self, objective_evaluator: ObjectiveEvaluate,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True):

        timer = OptimisationTimer(log=self.log, timeout=self.requirements.timeout)
        evaluator = EvaluationDispatcher(objective_evaluator, self.graph_generation_params.adapter,
                                         timer=timer, log=self.log, n_jobs=1)

        num_iter = 0
        best = Individual(self.initial_graph)
        evaluator([best])

        with timer as t:
            while not t.is_time_limit_reached(num_iter):
                new = mutation(types=self.change_types, ind=best, params=self.graph_generation_params,
                               requirements=self.requirements, log=self.log)
                evaluator([new])
                if new.fitness < best.fitness:
                    best = new
                num_iter += 1

        return best.graph


def run_with_random_search_composer():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    composer_params = {'available_operations': ['class_decompose', 'rf', 'linear', 'xgboost', 'dt'],
                       'optimizer': RandomSearchOptimizer}

    automl = Fedot(problem='classification', timeout=1, verbose_level=4,
                   preset='fast_train', composer_params=composer_params)

    automl.fit(train_data_path)
    automl.predict(test_data_path)
    print(automl.get_metrics())


if __name__ == '__main__':
    run_with_random_search_composer()
