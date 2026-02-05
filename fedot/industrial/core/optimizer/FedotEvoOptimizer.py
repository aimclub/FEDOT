from typing import Sequence

from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot.industrial.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY


class FedotEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 optimisation_params: dict = None):

        graph_optimizer_params = self._exclude_resample_from_mutations(graph_optimizer_params)
        self.mutation_agent_dict = {'random': RandomAgent,
                                    'bandit': MultiArmedBanditAgent,
                                    'contextual_bandit': ContextualMultiArmedBanditAgent,
                                    'neural_bandit': NeuralContextualMultiArmedBanditAgent}
        if optimisation_params is not None:
            graph_optimizer_params.adaptive_mutation_type = self._set_optimisation_strategy(graph_optimizer_params,
                                                                                            optimisation_params)
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        self.requirements = requirements
        # self.eval_dispatcher = IndustrialDispatcher(
        #     adapter=graph_generation_params.adapter,
        #     n_jobs=requirements.n_jobs,
        #     graph_cleanup_fn=_try_unfit_graph,
        #     delegate_evaluator=graph_generation_params.remote_evaluator)

    def _set_optimisation_strategy(self, graph_optimizer_params, optimisation_params):
        mutation_probs = FEDOT_MUTATION_STRATEGY[optimisation_params['mutation_strategy']]
        mutation_agent = self.mutation_agent_dict[optimisation_params['mutation_agent']]
        if optimisation_params['mutation_agent'].__contains__('random'):
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types,
                                            probs=mutation_probs)
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
