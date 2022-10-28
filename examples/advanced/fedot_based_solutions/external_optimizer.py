import logging
from random import choice
from typing import Any, Optional, Sequence, Union

from fedot.api.main import Fedot
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.gp_comp.evaluation import SimpleDispatcher
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, Mutation
from fedot.core.optimisers.objective import Objective, ObjectiveFunction
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.utils import fedot_project_root


class RandomMutationSearchOptimizer(GraphOptimizer):
    """
    Random search-based graph models optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Union[Graph, Sequence[Graph]] = (),
                 requirements: Optional[Any] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_parameters: Optional[GraphOptimizerParameters] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_parameters)
        self.mutation_types = [boosting_mutation, parameter_change_mutation,
                               MutationTypesEnum.single_edge,
                               MutationTypesEnum.single_change,
                               MutationTypesEnum.single_drop,
                               MutationTypesEnum.single_add]

    def optimise(self, objective: ObjectiveFunction):

        timer = OptimisationTimer(timeout=self.requirements.timeout)
        dispatcher = SimpleDispatcher(self.graph_generation_params.adapter)
        evaluator = dispatcher.dispatch(objective, timer)

        num_iter = 0

        initial_individuals = [Individual(graph) for graph in self.initial_graphs]
        best = choice(initial_individuals)
        evaluator([best])

        with timer as t:
            while not t.is_time_limit_reached(num_iter):
                mutation = Mutation(self.mutation_types, self.requirements, self.graph_generation_params)
                new = mutation(best)
                evaluator([new])
                if new.fitness < best.fitness:
                    best = new
                num_iter += 1

        return self.graph_generation_params.adapter.restore(best)


def run_with_random_search_composer():
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    composer_params = {'available_operations': ['class_decompose', 'rf', 'linear', 'xgboost', 'dt'],
                       'optimizer': RandomMutationSearchOptimizer}

    automl = Fedot(problem='classification', timeout=1, logging_level=logging.DEBUG,
                   preset='fast_train', **composer_params)

    automl.fit(train_data_path)
    automl.predict(test_data_path)
    print(automl.get_metrics())


if __name__ == '__main__':
    run_with_random_search_composer()
