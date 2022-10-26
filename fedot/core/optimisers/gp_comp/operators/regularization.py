from copy import deepcopy
from typing import TYPE_CHECKING

from fedot.core.dag.graph_utils import ordered_subnodes_hierarchy
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, EvaluationOperator, Operator
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


class Regularization(Operator):
    def __init__(self, parameters: 'GPGraphOptimizerParameters',
                 graph_generation_params: GraphGenerationParams):
        super().__init__(parameters=parameters)
        self.graph_generation_params = graph_generation_params

    def __call__(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        regularization_type = self.parameters.regularization_type
        if regularization_type is RegularizationTypesEnum.decremental:
            return self._decremental_regularization(population, evaluator)
        elif regularization_type is RegularizationTypesEnum.none:
            return population
        else:
            raise ValueError(f'Required regularization type not found: {regularization_type}')

    def _decremental_regularization(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        size = self.parameters.pop_size
        additional_inds = []
        prev_nodes_ids = set()
        for ind in population:
            prev_nodes_ids.add(ind.graph.descriptive_id)
            parent_operator = ParentOperator(type_='regularization',
                                             operators='decremental_regularization',
                                             parent_individuals=ind)
            subtree_inds = [Individual(OptGraph(deepcopy(ordered_subnodes_hierarchy(node))), parent_operator)
                            for node in ind.graph.nodes
                            if Regularization._is_fitted_subtree(self.graph_generation_params.adapter.restore(node))
                            and node.descriptive_id not in prev_nodes_ids]

            additional_inds.extend(subtree_inds)
            prev_nodes_ids.update(subtree.graph.root_node.descriptive_id for subtree in subtree_inds)

        additional_inds = [ind for ind in additional_inds if self.graph_generation_params.verifier(ind.graph)]
        evaluator(additional_inds)
        additional_inds.extend(population)
        if len(additional_inds) > size:
            additional_inds = sorted(additional_inds, key=lambda ind: ind.fitness)[:size]

        return additional_inds

    # TODO: remove this hack (e.g. provide smth like FitGraph with fit/unfit interface)
    @staticmethod
    def _is_fitted_subtree(node: OptNode) -> bool:
        return node.nodes_from and hasattr(node, 'fitted_operation') and node.fitted_operation
