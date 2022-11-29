from copy import deepcopy
from functools import partial
from random import choice, randint, random, sample
from typing import Callable, List, Union, Tuple, TYPE_CHECKING

import numpy as np

from fedot.core.adapter import register_native
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_utils import distance_to_root_level, ordered_subnodes_hierarchy, distance_to_primary_level
from fedot.core.optimisers.advisor import RemoveType
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.pipeline_advisor import check_for_specific_operations
from fedot.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


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
                 parameters: 'GPGraphOptimizerParameters',
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_generation_params

    def __call__(self, population: Union[Individual, PopulationT]) -> Union[Individual, PopulationT]:
        if isinstance(population, Individual):
            return self._mutation(population)
        return list(map(self._mutation, population))

    @staticmethod
    def get_mutation_prob(mut_id: MutationStrengthEnum, node: GraphNode,
                          default_mutation_prob: float = 0.7) -> float:
        """ Function returns mutation probability for certain node in the graph

        :param mut_id: MutationStrengthEnum mean weak or strong mutation
        :param node: root node of the graph
        :param default_mutation_prob: mutation probability used when mutation_id is invalid
        :return mutation_prob: mutation probability
        """
        if mut_id in list(MutationStrengthEnum):
            mutation_strength = mut_id.value
            mutation_prob = mutation_strength / (distance_to_primary_level(node) + 1)
        else:
            mutation_prob = default_mutation_prob
        return mutation_prob

    def _mutation(self, individual: Individual) -> Individual:
        """ Function applies mutation operator to graph """

        for _ in range(self.parameters.max_num_of_operator_attempts):
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
        """Apply mutation in several iterations with specific adaptation of each graph"""

        mutation_types = self.parameters.mutation_types
        is_static_mutation_type = random() < self.parameters.static_mutation_prob
        mutation_type = choice(mutation_types)
        mutation_names = []
        for _ in range(num_mut):
            # determine mutation type
            if not is_static_mutation_type:
                mutation_type = choice(mutation_types)
            is_custom_mutation = isinstance(mutation_type, Callable)

            if self._will_mutation_be_applied(mutation_type):
                # get the mutation function and adapt it
                mutation_func = self._get_mutation_func(mutation_type)
                new_graph = mutation_func(new_graph, requirements=self.requirements,
                                          params=self.graph_generation_params,
                                          opt_params=self.parameters)
                # log mutation
                mutation_names.append(str(mutation_type))
                if is_custom_mutation:
                    # custom mutation occurs once
                    break
        return new_graph, mutation_names

    def _will_mutation_be_applied(self, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return random() <= self.parameters.mutation_prob and mutation_type is not MutationTypesEnum.none

    @register_native
    def _simple_mutation(self, graph: OptGraph, **kwargs) -> OptGraph:
        """
        This type of mutation is passed over all nodes of the tree started from the root node and changes
        nodes’ operations with probability - 'node mutation probability'
        which is initialised inside the function

        :param graph: graph to mutate
        """

        exchange_node = self.graph_generation_params.node_factory.exchange_node
        visited_nodes = set()

        def replace_node_to_random_recursive(node: OptNode) -> OptGraph:
            if node not in visited_nodes and random() < node_mutation_probability:
                new_node = exchange_node(node)
                if new_node:
                    graph.update_node(node, new_node)
                # removed node must not be visited because it's outdated
                visited_nodes.add(node)
                # new node must not mutated if encountered further during traverse
                visited_nodes.add(new_node)
                for parent in node.nodes_from:
                    replace_node_to_random_recursive(parent)

        node_mutation_probability = self.get_mutation_prob(mut_id=self.parameters.mutation_strength,
                                                           node=graph.root_node)

        replace_node_to_random_recursive(graph.root_node)

        return graph

    @register_native
    def _single_edge_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        This mutation adds new edge between two random nodes in graph.

        :param graph: graph to mutate
        """
        old_graph = deepcopy(graph)

        for _ in range(self.parameters.max_num_of_operator_attempts):
            if len(graph.nodes) < 2 or graph.depth > self.requirements.max_depth:
                return graph

            source_node, target_node = sample(graph.nodes, 2)

            nodes_not_cycling = (target_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(source_node)])
            if nodes_not_cycling and (source_node not in target_node.nodes_from):
                graph.connect_nodes(source_node, target_node)
                break

        if graph.depth > self.requirements.max_depth:
            return old_graph
        return graph

    @register_native
    def _add_intermediate_node(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add between node and parent
        new_node = self.graph_generation_params.node_factory.get_parent_node(node_to_mutate, is_primary=False)
        if not new_node:
            return graph

        # rewire old children to new parent
        new_node.nodes_from = node_to_mutate.nodes_from
        node_to_mutate.nodes_from = [new_node]

        # add new node to graph
        graph.add_node(new_node)
        return graph

    @register_native
    def _add_separate_parent_node(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add as separate parent
        for iter_num in range(randint(1, 3)):
            new_node = self.graph_generation_params.node_factory.get_parent_node(node_to_mutate, is_primary=True)
            if not new_node:
                # there is no possible operators
                break
            if node_to_mutate.nodes_from:
                node_to_mutate.nodes_from.append(new_node)
            else:
                node_to_mutate.nodes_from = [new_node]
            graph.nodes.append(new_node)
        return graph

    @register_native
    def _add_as_child(self, graph: OptGraph, node_to_mutate: OptNode) -> OptGraph:
        # add as child
        old_node_children = graph.node_children(node_to_mutate)
        new_node_child = choice(old_node_children) if old_node_children else None
        operation_id = node_to_mutate.content['name']

        if check_for_specific_operations(operation_id):
            # data source, exog_ts and custom models moving is useless
            return graph

        new_node = self.graph_generation_params.node_factory.get_node(is_primary=False)
        if not new_node:
            return graph
        graph.add_node(new_node)
        graph.connect_nodes(node_parent=node_to_mutate, node_child=new_node)
        if new_node_child:
            graph.connect_nodes(node_parent=new_node, node_child=new_node_child)
            graph.disconnect_nodes(node_parent=node_to_mutate, node_child=new_node_child)

        return graph

    @register_native
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

    @register_native
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

    @register_native
    def _single_drop_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        """
        Drop single node from graph.

        :param graph: graph to mutate
        """
        node_to_del = choice(graph.nodes)
        node_name = node_to_del.content['name']
        removal_type = self.graph_generation_params.advisor.can_be_removed(node_to_del)
        # we can't delete all data source nodes
        if 'data_source' in node_name and self.only_one_data_source_node(graph):
            return graph
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

    @register_native
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
            max_depth = distance_to_primary_level(node_from_graph)
            is_primary_node_selected = (not node_from_graph.nodes_from) or (node_from_graph != graph.root_node and
                                                                            randint(0, 1))
        else:
            max_depth = self.requirements.max_depth - distance_to_root_level(graph, node_from_graph)
            is_primary_node_selected = \
                distance_to_root_level(graph, node_from_graph) >= self.requirements.max_depth and randint(0, 1)
        if is_primary_node_selected:
            new_subtree = self.graph_generation_params.node_factory.get_node(is_primary=True)
            if not new_subtree:
                return graph
        else:
            new_subtree = random_graph(self.graph_generation_params, self.requirements, max_depth).root_node
        graph.update_subtree(node_from_graph, new_subtree)
        return graph

    @register_native
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

    @register_native
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
            primary_node = self.graph_generation_params.node_factory.get_node(is_primary=True)
            if not primary_node:
                return graph
            graph.update_subtree(node_to_del, primary_node)
        return graph

    @register_native
    def _no_mutation(self, graph: OptGraph, *args, **kwargs) -> OptGraph:
        return graph

    def _get_mutation_func(self, mutation_type: Union[MutationTypesEnum, Callable]) -> Callable:
        if isinstance(mutation_type, Callable):
            mutation_func = mutation_type
        else:
            mutation_func = self.mutation_by_type(mutation_type)
        return self.graph_generation_params.adapter.adapt_func(mutation_func)

    def mutation_by_type(self, mutation_type: MutationTypesEnum) -> Callable:
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

    def only_one_data_source_node(self, graph: OptGraph):
        count = 0
        for node in graph.nodes:
            if 'data_source' in node.content['name']:
                if count == 1:
                    return False
                count += 1
        return count == 1
