from random import choice
from typing import Sequence, Optional, Dict, Any

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph
from pymonad.either import Either

from fedot.industrial.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot.industrial.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 optimisation_params: dict = {'mutation_agent': 'random',
                                              'mutation_strategy': 'params_mutation_strategy'}):
        graph_optimizer_params = self._init_industrial_optimizer_params(graph_optimizer_params, optimisation_params)
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        # self.operators.remove(self.crossover)
        self.evaluated_population = []
        self.requirements = requirements
        self.initial_graphs = initial_graphs
        self.eval_dispatcher = IndustrialDispatcher(
            adapter=graph_generation_params.adapter,
            n_jobs=requirements.n_jobs,
            graph_cleanup_fn=_try_unfit_graph,
            delegate_evaluator=graph_generation_params.remote_evaluator)

    def _init_industrial_optimizer_params(self, graph_optimizer_params, optimisation_params):
        self.mutation_agent_dict = {'random': RandomAgent,
                                    'bandit': MultiArmedBanditAgent,
                                    'contextual_bandit': ContextualMultiArmedBanditAgent,
                                    'neural_bandit': NeuralContextualMultiArmedBanditAgent}
        # Min pop size to avoid getting stuck in local maximum during optimization.
        self.min_pop_size = 10
        # Min reproduce attempt for evolve and mutation stage.
        self.min_reproduce_attempt = 50
        # Max number of evaluations attempts to create graph for next pop
        self.graph_generation_attempts = 100
        graph_optimizer_params = self._exclude_resample_from_mutations(graph_optimizer_params)
        graph_optimizer_params.adaptive_mutation_type = self._set_optimisation_strategy(graph_optimizer_params,
                                                                                        optimisation_params)
        return graph_optimizer_params

    def _set_optimisation_strategy(self, graph_optimizer_params, optimisation_params):
        self.optimisation_mutation_probs = FEDOT_MUTATION_STRATEGY[optimisation_params['mutation_strategy']]
        mutation_agent = self.mutation_agent_dict[optimisation_params['mutation_agent']]
        if optimisation_params['mutation_agent'].__contains__('random'):
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types,
                                            probs=self.optimisation_mutation_probs)
        else:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types)
        return mutation_agent

    def _exclude_resample_from_mutations(self, graph_optimizer_params):
        for mutation in graph_optimizer_params.mutation_types:
            try:
                is_invalid = mutation.__name__.__contains__('resample')
            except Exception:
                is_invalid = mutation.name.__contains__('resample')
            if is_invalid:
                graph_optimizer_params.mutation_types.remove(mutation)
        return graph_optimizer_params

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        pop_size = self.graph_optimizer_params.pop_size
        label = 'initial_assumptions'
        initial_individuals = [Individual(graph, metadata=self.requirements.static_individual_metadata)
                               for graph in self.initial_graphs]

        if len(initial_individuals) < pop_size:  # in case we have only one init assumption
            # change strategy of init assumption creation. Set max probability to node change mutation
            self.mutation.agent._probs = FEDOT_MUTATION_STRATEGY['initial_population_diversity_strategy']
            initial_individuals = self._extend_population(initial_individuals, pop_size)
            self.mutation.agent._probs = self.optimisation_mutation_probs
            label = 'extended_initial_assumptions'
        init_population = evaluator(initial_individuals)
        self._update_population(next_population=init_population, evaluator=evaluator, label=label)
        return init_population, evaluator

    def _extend_population(self, pop: PopulationT, target_pop_size: int, mutation_prob: list = None) -> PopulationT:
        verifier, new_population, new_ind = self.graph_generation_params.verifier, list(pop), 'empty'
        pop_graphs = [ind.graph for ind in new_population]
        for iter_num in range(self.graph_generation_attempts):
            for repr_attempt in range(self.min_reproduce_attempt):
                random_ind = choice(pop)
                new_ind = self.mutation(random_ind)
                if isinstance(new_ind, Individual):
                    # self.log.message(f'Successful mutation at attempt number: {repr_attempt}. '
                    #                  f'Obtain new pipeline - {new_ind.graph.descriptive_id}')
                    break
            is_valid_graph = verifier(new_ind.graph)
            is_new_graph = new_ind.graph not in pop_graphs
            if all([is_new_graph, is_valid_graph]):
                new_population.append(new_ind)
                pop_graphs.append(new_ind.graph)
            if len(new_population) == target_pop_size:
                break
        return new_population

    def _update_population(self,
                           next_population: PopulationT,
                           evaluator: EvaluationOperator = None,
                           label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        if self.requirements.keep_history:
            self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)
        self.population = next_population
        self.log.info(f'Generation num: {self.current_generation_num} size: {len(next_population)}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        if self.generations.stagnation_iter_count > 0:
            self.log.info(f'no improvements for {self.generations.stagnation_iter_count} iterations')
            self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')
        return next_population

    def _evolve_population(self,
                           population: PopulationT,
                           evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """

        def evolve_pop(population, evaluator):
            individuals_to_select = self.regularization(population, evaluator)
            new_population = self.reproducer.reproduce(individuals_to_select, evaluator)
            if self.reproducer.stop_condition or new_population is None:
                new_population = population
            else:
                self.log.message(f'Successful reproduction')

            # Adaptive agent experience collection & learning
            # Must be called after reproduction (that collects the new experience)
            experience = self.mutation.agent_experience
            experience.collect_results(new_population)
            self.mutation.agent.partial_fit(experience)

            # Use some part of previous pop in the next pop
            new_population = self.inheritance(population, new_population)
            new_population = self.elitism(self.generations.best_individuals, new_population)
            return new_population, evaluator

        return evolve_pop(population, evaluator)

    def get_structure_unique_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """ Increases structurally uniqueness of population to prevent stagnation in optimization process.
        Returned population may be not entirely unique, if the size of unique population is lower than MIN_POP_SIZE. """
        unique_population_with_ids = {ind.graph.descriptive_id: ind for ind in population}
        unique_population = list(unique_population_with_ids.values())
        is_population_too_small = len(unique_population) < self.min_pop_size
        # if size of unique population is too small, then extend it to MIN_POP_SIZE by repeating individuals
        if all([is_population_too_small, not self.reproducer.stop_condition]):
            self.mutation.agent._probs = FEDOT_MUTATION_STRATEGY['unique_population_strategy']
            unique_population = self._extend_population(pop=unique_population,
                                                        target_pop_size=self.min_pop_size)
            self.mutation.agent._probs = self.optimisation_mutation_probs
            population = evaluator(unique_population)
        return population, evaluator

    def _update_requirements(self, population, evaluator):
        # Defines adaptive changes to algorithm parameters like pop_size and operator probabilities
        if not self.generations.is_any_improved:
            self.graph_optimizer_params.mutation_prob, self.graph_optimizer_params.crossover_prob = \
                self._operators_prob.next(population)
            self.log.info(
                f'Next mutation proba: {self.graph_optimizer_params.mutation_prob}; '
                f'Next crossover proba: {self.graph_optimizer_params.crossover_prob}')
        self.graph_optimizer_params.pop_size = self._pop_size.next(population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(f'Next population size: {self.graph_optimizer_params.pop_size}; '
                      f''f'max graph depth: {self.requirements.max_depth}')

        # update requirements in operators
        for operator in self.operators:
            operator.update_requirements(self.graph_optimizer_params, self.requirements)
        return population, evaluator

    def _optimise_loop(self, population_to_eval, evaluator):
        evaluated_population = Either.insert((population_to_eval, evaluator)). \
            then(lambda opt_data: self._update_requirements(*opt_data)). \
            then(lambda opt_data: self._evolve_population(*opt_data)). \
            then(lambda fitness_data: self.get_structure_unique_population(*fitness_data)). \
            then(lambda reg_data: self._update_population(*reg_data)).value
        return evaluated_population

    def optimise(self, objective: ObjectiveFunction) -> Sequence[Graph]:
        with self.timer, self._progressbar as pbar:
            population_to_eval, evaluator = Either.insert(objective). \
                then(lambda objective: self.eval_dispatcher.dispatch(objective, self.timer)). \
                then(lambda evaluator: self._initial_population(evaluator)).value
            self.evaluated_population.append(population_to_eval)
            while not self.stop_optimization():
                population_to_eval = self._optimise_loop(population_to_eval=population_to_eval,
                                                         evaluator=evaluator)
                self.evaluated_population.append(population_to_eval)
                pbar.update()
            pbar.close()
        self._update_population(self.best_individuals, None, 'final_choices')
        best_models = [ind.graph for ind in self.best_individuals]
        return best_models
