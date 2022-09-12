from copy import deepcopy

from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, EvaluationOperator, Operator
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.node import Node
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class RegularizationTypesEnum(Enum):
    none = 'none'
    decremental = 'decremental'


class Regularization(Operator):
    def __init__(self, regularization_type: RegularizationTypesEnum,
                 requirements: PipelineComposerRequirements, graph_generation_params: GraphGenerationParams):
        self.regularization_type = regularization_type
        self.graph_generation_params = graph_generation_params
        self.requirements = requirements

    def __call__(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        if self.regularization_type is RegularizationTypesEnum.decremental:
            return self._decremental_regularization(population, evaluator)
        elif self.regularization_type is RegularizationTypesEnum.none:
            return population
        else:
            raise ValueError(f'Required regularization type not found: {self.regularization_type}')

    def update_requirements(self, new_requirements: PipelineComposerRequirements):
        self.requirements = new_requirements

    def _decremental_regularization(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        size = self.requirements.pop_size
        additional_inds = []
        prev_nodes_ids = set()
        for ind in population:
            prev_nodes_ids.add(ind.graph.root_node.descriptive_id)
            parent_operator = ParentOperator(type_='regularization',
                                             operators='decremental_regularization',
                                             parent_individuals=ind)
            subtree_inds = [Individual(OptGraph(deepcopy(node.ordered_subnodes_hierarchy())), parent_operator)
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

    @staticmethod
    def _is_fitted_subtree(node: Node) -> bool:
        return node.nodes_from and node.fitted_operation
