from copy import deepcopy
from functools import partial
from random import choice, randint, random, sample
from typing import Callable, List, Union, Tuple, Sequence

import numpy as np

from fedot.core.composer.advisor import RemoveType
from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual, ParentOperator
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.utilities.data_structures import ComparableEnum as Enum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements, \
    MutationStrengthEnum


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    single_add = 'single_add',
    single_change = 'single_change',
    single_drop = 'single_drop',
    single_edge = 'single_edge'

    none = 'none'


class Mutation(Operator):
    def __init__(self,
                 parameters: 'GPGraphGenerationParameters',
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 # TODO: move these 2 to gp_parameters
                 max_num_of_mutation_attempts: int = 100,
                 static_mutation_probability: float = 0.7):
        super().__init__(parameters, requirements)
        self.mutation_types = parameters.mutation_types
        self.graph_generation_params = graph_generation_params
        self.max_num_of_mutation_attempts = max_num_of_mutation_attempts
        self.static_mutation_probability = static_mutation_probability

    def __call__(self, population: Union[Individual, PopulationT]) -> Union[Individual, PopulationT]:
        if isinstance(population, Individual):
            return self._mutation(population)
        return list(map(self._mutation, population))

    @staticmethod
    def get_mutation_prob(mut_id: MutationStrengthEnum, node: GraphNode) -> float:
        """ Function returns mutation probability for certain node in the graph

        :param mut_id: MutationStrengthEnum mean weak or strong mutation
        :param node: root node of the graph
        :return mutation_prob: mutation probability
        """

        default_mutation_prob = 0.7  # TODO: why it duplicates parameters.mutation_prob?
        if mut_id in list(MutationStrengthEnum):
            mutation_strength = mut_id.value
            mutation_prob = mutation_strength / (node.distance_to_primary_level + 1)
        else:
            mutation_prob = default_mutation_prob
        return mutation_prob

    def _mutation(self, individual: Individual) -> Individual:
        """ Function applies mutation operator to graph """

        for _ in range(self.max_num_of_mutation_attempts):
            new_graph = deepcopy(individual.graph)
            num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)

            new_graph, mutation_names = self._adapt_and_apply_mutations(new_graph, num_mut)

            is_correct_graph = self.graph_generation_params.verifier(new_graph)
            if is_correct_graph:
                parent_operator = ParentOperator(type_='mutation', operators=tuple(mutation_names),
                                                 parent_individuals=individual)
                return Individual(new_graph, parent_operator)

        self.log.debug('Number of mutation attempts exceeded. '
                       'Please check composer requirements for correctness.')

        return individual

    def _adapt_and_apply_mutations(self, new_graph: OptGraph, num_mut: int) -> Tuple[OptGraph, List[str]]:
        """
        Apply mutation in several iterations with specific adaptation of each graph
        """

        is_static_mutation_type = random() < self.static_mutation_probability
        static_mutation_type = choice(self.mutation_types)
        mutation_names = []
        for _ in range(num_mut):
            mutation_type = static_mutation_type \
                if is_static_mutation_type else choice(self.mutation_types)
            is_custom_mutation = isinstance(mutation_type, Callable)

            if is_custom_mutation:
                new_graph = self.graph_generation_params.adapter.restore(new_graph)
            else:
                if not isinstance(new_graph, OptGraph):
                    new_graph = self.graph_generation_params.adapter.adapt(new_graph)
            new_graph = self._apply_mutation(new_graph, mutation_type, is_custom_mutation)
            mutation_names.append(str(mutation_type))
            if not isinstance(new_graph, OptGraph):
                new_graph = self.graph_generation_params.adapter.adapt(new_graph)
            if is_custom_mutation:
                # custom mutation occurs once
                break
        return new_graph, mutation_names

    def _apply_mutation(self, new_graph: Union[Graph, OptGraph], mutation_type: Union[MutationTypesEnum, Callable],
                        is_custom_mutation: bool) -> Union[Graph, OptGraph]:
        """
          Apply mutation for adapted graph
        """
        if self._will_mutation_be_applied(self.requirements.mutation_prob, mutation_type):
            if is_custom_mutation:
                mutation_func = mutation_type
            else:
                mutation_func = self.mutation_by_type(mutation_type)
            graph_copy = deepcopy(new_graph)
            new_graph = mutation_func(new_graph, requirements=self.requirements,
                                      params=self.graph_generation_params)
            if not new_graph.nodes:
                return graph_copy
        return new_graph

    @staticmethod
    def _will_mutation_be_applied(mutation_prob: float, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return not (random() > mutation_prob or mutation_type is MutationTypesEnum.none)

    def _simple_mutation(self, graph: OptGraph, **kwargs) -> OptGraph:
        """
        This type of mutation is passed over all nodes of the tree started from the root node and changes
        nodes’ operations with probability - 'node mutation probability'
        which is initialised inside the function

        :param graph: graph to mutate
        """

        def replace_node_to_random_recursive(node: OptGraph) -> OptGraph:
            if random() < node_mutation_probability:
                new_node = self.graph_generation_params.node_factory.exchange_node(node)
                if new_node:
                    graph.update_node(node, new_node)
                if node.nodes_from:
                    for parent in node.nodes_from:
                        replace_node_to_random_recursive(parent)

        node_mutation_probability = self.get_mutation_prob(mut_id=self.requirements.mutation_strength,
                                                           node=graph.root_node)

        replace_node_to_random_recursive(graph.root_node)

        return graph

    def _single_edge_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        This mutation adds new edge between two random nodes in graph.

        :param graph: graph to mutate
        """
        old_graph = deepcopy(graph)

        for _ in range(self.max_num_of_mutation_attempts):
            if len(graph.nodes) < 2 or graph.depth > self.requirements.max_depth:
                return graph

            source_node, target_node = sample(graph.nodes, 2)

            nodes_not_cycling = (target_node.descriptive_id not in
                                 [n.descriptive_id for n in source_node.ordered_subnodes_hierarchy()])
            if nodes_not_cycling and (source_node not in target_node.nodes_from):
                graph.connect_nodes(source_node, target_node)
                break

        if graph.depth > self.requirements.max_depth:
            return old_graph
        return graph

    def _add_intermediate_node(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add between node and parent
        new_node = self.graph_generation_params.node_factory.get_parent_node(node_to_mutate, primary=False)
        if not new_node:
            return graph
        new_node.nodes_from = node_to_mutate.nodes_from
        node_to_mutate.nodes_from = [new_node]
        graph.nodes.append(new_node)
        return graph

    def _add_separate_parent_node(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add as separate parent
        for iter_num in range(randint(1, 3)):
            new_node = self.graph_generation_params.node_factory.get_parent_node(node_to_mutate, primary=True)
            if not new_node:
                # there is no possible operators
                break
            if node_to_mutate.nodes_from:
                node_to_mutate.nodes_from.append(new_node)
            else:
                node_to_mutate.nodes_from = [new_node]
            graph.nodes.append(new_node)
        return graph

    def _add_as_child(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add as child
        new_node = self.graph_generation_params.node_factory.get_node(primary=False)
        if not new_node:
            return graph
        parents_node_to_mutate = node_to_mutate.nodes_from or []
        graph.update_node(old_node=node_to_mutate, new_node=new_node)
        graph.add_node(node_to_mutate)
        graph.connect_nodes(node_parent=node_to_mutate, node_child=new_node)
        for node_parent in parents_node_to_mutate:
            graph.disconnect_nodes(node_parent=node_parent, node_child=new_node)
        return graph

    def _single_add_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        Add new node between two sequential existing modes

        :param graph: graph to mutate
        """

        if graph.depth >= self.requirements.max_depth:
            # add mutation is not possible
            return graph

        node_to_mutate = choice(graph.nodes)

        single_add_strategies = [self._add_as_child, self._add_separate_parent_node]
        if node_to_mutate.nodes_from:
            single_add_strategies.append(self._add_intermediate_node)
        strategy = choice(single_add_strategies)

        result = strategy(graph, node_to_mutate)
        return result

    def _single_change_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        Change node between two sequential existing modes.

        :param graph: graph to mutate
        """
        node = choice(graph.nodes)
        new_node = self.graph_generation_params.node_factory.exchange_node(node)
        if not new_node:
            return graph
        graph.update_node(node, new_node)
        return graph

    def _single_drop_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        Drop single node from graph.

        :param graph: graph to mutate
        """
        node_to_del = choice(graph.nodes)
        node_name = node_to_del.content['name']
        removal_type = self.graph_generation_params.advisor.can_be_removed(str(node_name))
        if removal_type == RemoveType.with_direct_children:
            # TODO refactor workaround with data_source
            nodes_to_delete = \
                [n for n in graph.nodes if str(node_name) in n.descriptive_id and
                 n.descriptive_id.count('data_source') == 1]
            for child_node in nodes_to_delete:
                graph.delete_node(child_node)
            graph.delete_node(node_to_del)
        elif removal_type == RemoveType.with_parents:
            graph.delete_subtree(node_to_del)
        elif removal_type != RemoveType.forbidden:
            graph.delete_node(node_to_del)
            if node_to_del.nodes_from:
                children = graph.node_children(node_to_del)
                for child in children:
                    if child.nodes_from:
                        child.nodes_from.extend(node_to_del.nodes_from)
                    else:
                        child.nodes_from = node_to_del.nodes_from
        return graph

    def _tree_growth(self, graph: OptGraph, local_growth: bool = True) -> OptGraph:
        """
        This mutation selects a random node in a tree, generates new subtree,
        and replaces the selected node's subtree.

        :param graph: graph to mutate
        :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
        selected random node, if false then previous depth of selected node doesn't affect to
        new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
        """
        node_from_graph = choice(graph.nodes)
        if local_growth:
            max_depth = node_from_graph.distance_to_primary_level
            is_primary_node_selected = (not node_from_graph.nodes_from) or (node_from_graph != graph.root_node and
                                                                            randint(0, 1))
        else:
            max_depth = self.requirements.max_depth - graph.distance_to_root_level(node_from_graph)
            is_primary_node_selected = \
                graph.distance_to_root_level(node_from_graph) >= self.requirements.max_depth and randint(0, 1)
        if is_primary_node_selected:
            new_subtree = self.graph_generation_params.node_factory.get_node(primary=True)
            if not new_subtree:
                return graph
        else:
            new_subtree = random_graph(self.graph_generation_params, self.requirements, max_depth).root_node
        graph.update_subtree(node_from_graph, new_subtree)
        return graph

    def _growth_mutation(self, graph: OptGraph, local_growth: bool = True, **kwargs) -> OptGraph:
        """
        This mutation adds new nodes to the graph (just single node between existing nodes or new subtree).

        :param graph: graph to mutate
        :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
        selected random node, if false then previous depth of selected node doesn't affect to
        new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
        """

        if random() > 0.5:
            # simple growth (one node can be added)
            return self._single_add_mutation(graph, self.requirements.max_depth)
        else:
            # advanced growth (several nodes can be added)
            return self._tree_growth(graph, local_growth)

    def _reduce_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        Selects a random node in a tree, then removes its subtree. If the current arity of the node's
        parent is more than the specified minimal arity, then the selected node is also removed.
        Otherwise, it is replaced by a random primary node.

        :param graph: graph to mutate
        """
        if len(graph.nodes) == 1:
            return graph

        nodes = [node for node in graph.nodes if node is not graph.root_node]
        node_to_del = choice(nodes)
        children = graph.node_children(node_to_del)
        is_possible_to_delete = all([len(child.nodes_from) - 1 >= self.requirements.min_arity for child in children])
        if is_possible_to_delete:
            graph.delete_subtree(node_to_del)
        else:
            primary_node = self.graph_generation_params.node_factory.get_node(primary=True)
            if not primary_node:
                return graph
            graph.update_subtree(node_to_del, primary_node)
        return graph

    def _no_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        return graph

    def mutation_by_type(self, mutation_type: MutationTypesEnum):
        mutations = {
            MutationTypesEnum.none: self._no_mutation,
            MutationTypesEnum.simple: self._simple_mutation,
            MutationTypesEnum.growth: partial(self._growth_mutation, local_growth=False),
            MutationTypesEnum.local_growth: partial(self._growth_mutation, local_growth=True),
            MutationTypesEnum.reduce: self._reduce_mutation,
            MutationTypesEnum.single_add: self._single_add_mutation,
            MutationTypesEnum.single_edge: self._single_edge_mutation,
            MutationTypesEnum.single_drop: self._single_drop_mutation,
            MutationTypesEnum.single_change: self._single_change_mutation,
        }
        if mutation_type in mutations:
            return mutations[mutation_type]
        else:
            raise ValueError(f'Required mutation type is not found: {mutation_type}')
