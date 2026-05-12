import itertools
from copy import deepcopy
from itertools import chain
from math import ceil
from random import choice, sample, random
from typing import Sequence, Optional
from typing import Tuple

import numpy as np
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.hyperparams import ParametersChanger
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import graph_has_cycle, distance_to_primary_level
from golem.core.dag.graph_utils import node_depth, nodes_from_layer
from golem.core.dag.verification_rules import ERROR_PREFIX
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.genetic.gp_operators import equivalent_subtree, replace_subtrees
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import AlgorithmParameters
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.utilities.data_structures import ComparableEnum as Enum
from golem.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.industrial.core.repository.excluded import EXCLUDED_OPERATION_MUTATION, TEMPORARY_EXCLUDED
from fedot.industrial.core.repository.model_repository import default_industrial_availiable_operation, \
    PRIMARY_FORECASTING_MODELS


class MutationStrengthEnumIndustrial(Enum):
    weak = 1.0
    mean = 3.0
    strong = 5.0


def get_mutation_prob(mut_id: MutationStrengthEnumIndustrial, node: Optional[GraphNode],
                      default_mutation_prob: float = 0.7) -> float:
    """ Function returns mutation probability for certain node in the graph

    :param mut_id: MutationStrengthEnum mean weak or strong mutation
    :param node: root node of the graph
    :param default_mutation_prob: mutation probability used when mutation_id is invalid or graph has cycles
    :return mutation_prob: mutation probability
    """
    graph_cycled = True if node is None else distance_to_primary_level(
        node) < 0
    correct_graph = mut_id in list(
        MutationStrengthEnumIndustrial) and not graph_cycled
    mutation_prob = mut_id.value / \
        (distance_to_primary_level(node) +
         1) if correct_graph else default_mutation_prob
    return mutation_prob


class IndustrialMutations:
    def __init__(self, task_type):
        self.node_adapter = PipelineAdapter()
        self.task_type = Task(task_type)
        self.excluded_mutation = EXCLUDED_OPERATION_MUTATION[self.task_type.task_type.value]
        self.industrial_data_operations = default_industrial_availiable_operation(
            self.task_type.task_type.value)
        self._define_operation_space()
        self._define_basis_and_extractor_space()

    def _define_operation_space(self):
        self.excluded = [list(TEMPORARY_EXCLUDED[x].keys())
                         for x in TEMPORARY_EXCLUDED.keys()]
        self.excluded = (list(itertools.chain(*self.excluded)))
        self.excluded = self.excluded + self.excluded_mutation
        self.industrial_data_operations = [
            operation for operation in self.industrial_data_operations if operation not in self.excluded]
        self.primary_models = {'ts_forecasting': PRIMARY_FORECASTING_MODELS}

    def _define_basis_and_extractor_space(self):
        self.basis_models = get_operations_for_task(
            task=self.task_type, mode='data_operation', tags=["basis"])
        self.ts_preproc_model = get_operations_for_task(task=self.task_type, mode='data_operation',
                                                        tags=["smoothing",
                                                              # "non_lagged"
                                                              ])
        self.ts_model = get_operations_for_task(task=self.task_type, mode='model',
                                                tags=["time_series"])
        self.nn_ts_model = get_operations_for_task(task=self.task_type, mode='model',
                                                   tags=["fedot_NN_forecasting"])
        self.ts_model = self.ts_model + self.nn_ts_model
        self.ts_model = [x for x in self.ts_model if x not in self.excluded]
        extractors = get_operations_for_task(
            task=self.task_type, mode='data_operation', tags=["extractor"])
        self.extractors = [
            x for x in extractors if x in self.industrial_data_operations and x != 'channel_filtration']

    def transform_to_pipeline_node(self, node):
        return self.node_adapter._transform_to_pipeline_node(node)

    def transform_to_opt_node(self, node):
        return self.node_adapter._transform_to_opt_node(node)

    def parameter_change_mutation(self,
                                  pipeline: Pipeline, requirements, graph_gen_params, parameters, **kwargs) -> Pipeline:
        """
        This type of mutation is passed over all nodes and changes
        hyperparameters of the operations with probability - 'node mutation probability'
        which is initialised inside the function
        """
        node_mutation_probability = get_mutation_prob(mut_id=parameters.mutation_strength,
                                                      node=pipeline.root_node)
        for node in pipeline.nodes:
            lagged = node.operation.metadata.id in (
                'lagged', 'sparse_lagged', 'exog_ts')
            do_mutation = random() < (node_mutation_probability * (0.5 if lagged else 1))
            if do_mutation:
                operation_name = node.operation.operation_type
                current_params = node.parameters

                # Perform specific change for particular parameter
                changer = ParametersChanger(operation_name, current_params)
                try:
                    new_params = changer.get_new_operation_params()
                    if new_params is not None:
                        node.parameters = new_params
                except Exception as ex:
                    pipeline.log.error(ex)
        return pipeline

    def single_edge_mutation(self,
                             graph: OptGraph,
                             requirements: GraphRequirements,
                             graph_gen_params: GraphGenerationParams,
                             parameters: GPAlgorithmParameters
                             ) -> OptGraph:
        """
        This mutation adds new edge between two random nodes in graph.

        :param graph: graph to mutate
        """

        def nodes_not_cycling(source_node: OptNode, target_node: OptNode):
            parents = source_node.nodes_from
            while parents:
                if target_node not in parents:
                    grandparents = []
                    for parent in parents:
                        grandparents.extend(parent.nodes_from)
                    parents = grandparents
                else:
                    return False
            return True

        for _ in range(parameters.max_num_of_operator_attempts):
            if len(graph.nodes) < 2 or graph.depth > requirements.max_depth:
                return graph

            source_node, target_node = sample(graph.nodes, 2)
            if source_node not in target_node.nodes_from:
                if graph_has_cycle(graph):
                    graph.connect_nodes(source_node, target_node)
                    break
                else:
                    if nodes_not_cycling(source_node, target_node):
                        graph.connect_nodes(source_node, target_node)
                        break
        return graph

    def add_intermediate_node(self,
                              graph: OptGraph,
                              node_to_mutate: OptNode,
                              node_factory: OptNodeFactory) -> OptGraph:
        # add between node and parent
        new_node = node_factory.get_parent_node(
            self.transform_to_opt_node(node_to_mutate), is_primary=False)
        new_node = self.transform_to_pipeline_node(new_node)
        if not new_node:
            return graph

        # rewire old children to new parent
        new_node.nodes_from = node_to_mutate.nodes_from
        node_to_mutate.nodes_from = [new_node]

        # add new node to graph
        graph.add_node(new_node)
        return graph

    def add_separate_parent_node(self,
                                 graph: OptGraph,
                                 node_to_mutate: PipelineNode,
                                 node_factory: OptNodeFactory) -> OptGraph:
        # add as separate parent
        new_node = node_factory.get_parent_node(
            self.transform_to_opt_node(node_to_mutate), is_primary=True)
        if not new_node:
            # there is no possible operators
            return graph
        new_node = self.transform_to_pipeline_node(new_node)
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(new_node)
        else:
            node_to_mutate.nodes_from = [new_node]
        graph.nodes.append(new_node)
        return graph

    def add_as_child(self,
                     graph: OptGraph,
                     node_to_mutate: OptNode,
                     node_factory: OptNodeFactory) -> OptGraph:
        # add as child
        old_node_children = graph.node_children(node_to_mutate)
        new_node_child = choice(
            old_node_children) if old_node_children else None

        while True:
            new_node = node_factory.get_node(is_primary=False)
            if new_node.name in self.industrial_data_operations:
                break
        if not new_node:
            return graph

        new_node = self.transform_to_pipeline_node(new_node)

        if graph.depth == 1:
            graph.add_node(new_node)
            graph.connect_nodes(node_parent=new_node,
                                node_child=node_to_mutate)
        else:
            graph.add_node(new_node)
            graph.connect_nodes(node_parent=node_to_mutate,
                                node_child=new_node)

        if new_node_child:
            graph.connect_nodes(node_parent=new_node,
                                node_child=new_node_child)
            graph.disconnect_nodes(
                node_parent=node_to_mutate,
                node_child=new_node_child,
                clean_up_leftovers=True)

        return graph

    def single_add(self,
                   graph: OptGraph,
                   requirements: GraphRequirements,
                   graph_gen_params: GraphGenerationParams,
                   parameters: AlgorithmParameters
                   ) -> OptGraph:
        """
        Add new node between two sequential existing modes

        :param graph: graph to mutate
        """

        if graph.depth >= requirements.max_depth:
            # add mutation is not possible
            return graph

        node_to_mutate = choice(graph.nodes)

        single_add_strategies = [
            self.add_as_child,
            self.add_separate_parent_node
        ]
        if node_to_mutate.nodes_from:
            single_add_strategies.append(self.add_intermediate_node)
        strategy = choice(single_add_strategies)

        result = strategy(graph, node_to_mutate, graph_gen_params.node_factory)
        return result

    def single_change(self,
                      graph: OptGraph,
                      requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: AlgorithmParameters
                      ) -> OptGraph:
        """
        Change node between two sequential existing modes.

        :param graph: graph to mutate
        """
        node = choice(graph.nodes)
        task = graph_gen_params.advisor.task.task_type.name
        if task in self.primary_models.keys():
            graph_gen_params.node_factory.graph_model_repository.operations_by_keys['primary'] \
                = self.primary_models[task]
        new_node = graph_gen_params.node_factory.exchange_node(
            self.transform_to_opt_node(node))
        if not new_node:
            return graph
        graph.update_node(node, self.transform_to_pipeline_node(new_node))
        return graph

    def single_drop(self,
                    graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters
                    ) -> OptGraph:
        """
        Drop single node from graph.

        :param graph: graph to mutate
        """
        if len(graph.nodes) < 2:
            return graph
        node_to_del = choice(graph.nodes)
        node_name = node_to_del.name
        node_to_del = self.transform_to_opt_node(node_to_del)
        removal_type = graph_gen_params.advisor.can_be_removed(node_to_del)
        if removal_type == RemoveType.with_direct_children:
            # TODO refactor workaround with data_source
            graph.delete_node(node_to_del)
            nodes_to_delete = [n for n in graph.nodes if n.descriptive_id.count(
                'data_source') == 1 and node_name in n.descriptive_id]
            for child_node in nodes_to_delete:
                graph.delete_node(child_node, reconnect=ReconnectType.all)
        elif removal_type == RemoveType.with_parents:
            graph.delete_subtree(node_to_del)
        elif removal_type == RemoveType.node_rewire:
            try:
                graph.delete_node(node_to_del, reconnect=ReconnectType.all)
            except Exception:
                _ = 1
        elif removal_type == RemoveType.node_only:
            graph.delete_node(node_to_del, reconnect=ReconnectType.none)
        elif removal_type == RemoveType.forbidden:
            pass
        else:
            raise ValueError(
                "Unknown advice (RemoveType) returned by Advisor ")
        return graph

    def add_preprocessing(self,
                          graph: Pipeline, **kwargs) -> Pipeline:

        # create subtree with basis transformation and feature extraction
        transformation_node = PipelineNode(choice(self.basis_models))
        node_to_add_transformation = list(
            filter(lambda x: x.name in self.extractors, graph.nodes))
        if len(node_to_add_transformation) > 0:
            node_to_add_transformation = node_to_add_transformation[0]
            mutation_node = PipelineNode(
                node_to_add_transformation.name, nodes_from=[transformation_node])
            graph.update_node(
                old_node=node_to_add_transformation, new_node=mutation_node)
        return graph

    def __add_forecasting_preprocessing(self,
                                        graph: Pipeline, **kwargs) -> Pipeline:

        # create subtree with basis transformation and feature extraction
        transformation_node = PipelineNode(choice(self.ts_preproc_model))
        node_to_add_transformation = list(
            filter(lambda x: x.name in self.ts_model, graph.nodes))[0]
        mutation_node = PipelineNode(
            node_to_add_transformation.name, nodes_from=[transformation_node])
        graph.update_node(old_node=node_to_add_transformation,
                          new_node=mutation_node)
        return graph

    def add_forecasting_preprocessing(self,
                                      graph: Pipeline, **kwargs) -> Pipeline:
        mutation_dict = {'lagged_mutation': self.add_lagged,
                         'preproc_mutation': self.__add_forecasting_preprocessing}
        type_of_mutation = np.random.choice(
            ['lagged_mutation', 'preproc_mutation'])
        return mutation_dict[type_of_mutation](graph)

    def add_lagged(self, pipeline: Pipeline, **kwargs) -> Pipeline:
        lagged = ['lagged_forecaster']
        current_operation = list(reversed([x.name for x in pipeline.nodes]))
        if 'lagged_forecaster' in current_operation:
            return pipeline
        else:
            pipeline = PipelineBuilder().add_sequence(
                *lagged,
                branch_idx=0).add_sequence(
                *current_operation,
                branch_idx=1).join_branches('bagging').build()
            return pipeline


def _get_default_industrial_mutations(
        task_type: TaskTypesEnum,
        params) -> Sequence[MutationTypesEnum]:
    ind_mutations = IndustrialMutations(task_type=task_type)
    mutations = [
        ind_mutations.parameter_change_mutation,
        ind_mutations.single_change,
        ind_mutations.add_preprocessing,
        ind_mutations.single_drop,
        ind_mutations.single_add

    ]
    # TODO remove workaround after boosting mutation fix
    if task_type == TaskTypesEnum.ts_forecasting:
        mutations = [
            ind_mutations.parameter_change_mutation,
            ind_mutations.single_change,
            ind_mutations.add_forecasting_preprocessing,
            ind_mutations.single_drop,
            ind_mutations.single_add
        ]
        # #mutations.append(boosting_mutation)
        # mutations.append(ind_mutations.add_lagged)
        # mutations.remove(ind_mutations.add_preprocessing)
    return mutations


class IndustrialCrossover:
    @register_native
    def subtree_crossover(self,
                          graph_1: OptGraph,
                          graph_2: OptGraph,
                          max_depth: int,
                          inplace: bool = True) -> Tuple[OptGraph,
                                                         OptGraph]:
        """Performed by the replacement of random subtree
        in first selected parent to random subtree from the second parent"""

        if not inplace:
            graph_1 = deepcopy(graph_1)
            graph_2 = deepcopy(graph_2)
        else:
            graph_1 = graph_1
            graph_2 = graph_2

        random_layer_in_graph_first = choice(range(graph_1.depth))
        min_second_layer = 1 if random_layer_in_graph_first == 0 and graph_2.depth > 1 else 0
        random_layer_in_graph_second = choice(
            range(min_second_layer, graph_2.depth))

        node_from_graph_first = choice(nodes_from_layer(
            graph_1, random_layer_in_graph_first))
        node_from_graph_second = choice(nodes_from_layer(
            graph_2, random_layer_in_graph_second))

        replace_subtrees(
            graph_1,
            graph_2,
            node_from_graph_first,
            node_from_graph_second,
            random_layer_in_graph_first,
            random_layer_in_graph_second,
            max_depth)

        return graph_1, graph_2

    @register_native
    def one_point_crossover(self,
                            graph_first: OptGraph,
                            graph_second: OptGraph,
                            max_depth: int) -> Tuple[OptGraph, OptGraph]:
        """Finds common structural parts between two trees, and after that randomly
        chooses the location of nodes, subtrees of which will be swapped"""
        pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
        if pairs_of_nodes:
            node_from_graph_first, node_from_graph_second = choice(
                pairs_of_nodes)

            layer_in_graph_first = graph_first.depth - \
                node_depth(node_from_graph_first)
            layer_in_graph_second = graph_second.depth - \
                node_depth(node_from_graph_second)

            replace_subtrees(
                graph_first,
                graph_second,
                node_from_graph_first,
                node_from_graph_second,
                layer_in_graph_first,
                layer_in_graph_second,
                max_depth)
        return graph_first, graph_second

    @register_native
    def exchange_edges_crossover(self,
                                 graph_first: OptGraph,
                                 graph_second: OptGraph,
                                 max_depth):
        """Parents exchange a certain number of edges with each other. The number of
        edges is defined as half of the minimum number of edges of both parents, rounded up"""

        def find_edges_in_other_graph(edges, graph: OptGraph):
            new_edges = []
            for parent, child in edges:
                parent_new = graph.get_nodes_by_name(str(parent))
                if parent_new:
                    parent_new = parent_new[0]
                else:
                    parent_new = OptNode(str(parent))
                    graph.add_node(parent_new)
                child_new = graph.get_nodes_by_name(str(child))
                if child_new:
                    child_new = child_new[0]
                else:
                    child_new = OptNode(str(child))
                    graph.add_node(child_new)
                new_edges.append((parent_new, child_new))
            return new_edges

        edges_1 = graph_first.get_edges()
        edges_2 = graph_second.get_edges()
        count = ceil(min(len(edges_1), len(edges_2)) / 2)
        choice_edges_1 = sample(edges_1, count)
        choice_edges_2 = sample(edges_2, count)

        for parent, child in choice_edges_1:
            child.nodes_from.remove(parent)
        for parent, child in choice_edges_2:
            child.nodes_from.remove(parent)

        old_edges1 = graph_first.get_edges()
        old_edges2 = graph_second.get_edges()

        new_edges_2 = find_edges_in_other_graph(choice_edges_1, graph_second)
        new_edges_1 = find_edges_in_other_graph(choice_edges_2, graph_first)

        for parent, child in new_edges_1:
            if (parent, child) not in old_edges1:
                child.nodes_from.append(parent)
        for parent, child in new_edges_2:
            if (parent, child) not in old_edges2:
                child.nodes_from.append(parent)

        return graph_first, graph_second

    @register_native
    def exchange_parents_one_crossover(
            self,
            graph_first: OptGraph,
            graph_second: OptGraph,
            max_depth: int):
        """For the selected node for the first parent, change the parent nodes to
        the parent nodes of the same node of the second parent. Thus, the first child is obtained.
        The second child is a copy of the second parent"""

        def find_nodes_in_other_graph(nodes, graph: OptGraph):
            new_nodes = []
            for node in nodes:
                new_node = graph.get_nodes_by_name(str(node))
                if new_node:
                    new_node = new_node[0]
                else:
                    new_node = OptNode(str(node))
                    graph.add_node(new_node)
                new_nodes.append(new_node)
            return new_nodes

        edges = graph_second.get_edges()
        nodes_with_parent_or_child = list(set(chain(*edges)))
        if nodes_with_parent_or_child:

            selected_node = choice(nodes_with_parent_or_child)
            parents = selected_node.nodes_from

            node_from_first_graph = find_nodes_in_other_graph(
                [selected_node], graph_first)[0]

            node_from_first_graph.nodes_from = []
            old_edges1 = graph_first.get_edges()

            if parents:
                parents_in_first_graph = find_nodes_in_other_graph(
                    parents, graph_first)
                for parent in parents_in_first_graph:
                    if (parent, node_from_first_graph) not in old_edges1:
                        node_from_first_graph.nodes_from.append(parent)

        return graph_first, graph_second

    @register_native
    def exchange_parents_both_crossover(
            self,
            graph_first: OptGraph,
            graph_second: OptGraph,
            max_depth: int):
        """For the selected node for the first parent, change the parent nodes to
        the parent nodes of the same node of the second parent. Thus, the first child is obtained.
        The second child is formed in a similar way"""

        parents_in_first_graph = []
        parents_in_second_graph = []

        def find_nodes_in_other_graph(nodes, graph: OptGraph):
            new_nodes = []
            for node in nodes:
                new_node = graph.get_nodes_by_name(str(node))
                if new_node:
                    new_node = new_node[0]
                else:
                    new_node = OptNode(str(node))
                    graph.add_node(new_node)
                new_nodes.append(new_node)
            return new_nodes

        edges = graph_second.get_edges()
        nodes_with_parent_or_child = list(set(chain(*edges)))
        if nodes_with_parent_or_child:

            selected_node2 = choice(nodes_with_parent_or_child)
            parents2 = selected_node2.nodes_from
            if parents2:
                parents_in_first_graph = find_nodes_in_other_graph(
                    parents2, graph_first)

            selected_node1 = find_nodes_in_other_graph(
                [selected_node2], graph_first)[0]
            parents1 = selected_node1.nodes_from
            if parents1:
                parents_in_second_graph = find_nodes_in_other_graph(
                    parents1, graph_second)

            for p in parents1:
                selected_node1.nodes_from.remove(p)
            for p in parents2:
                selected_node2.nodes_from.remove(p)

            old_edges1 = graph_first.get_edges()
            old_edges2 = graph_second.get_edges()

            for parent in parents_in_first_graph:
                if (parent, selected_node1) not in old_edges1:
                    selected_node1.nodes_from.append(parent)

            for parent in parents_in_second_graph:
                if (parent, selected_node2) not in old_edges2:
                    selected_node2.nodes_from.append(parent)

        return graph_first, graph_second


def has_no_data_flow_conflicts_in_industrial_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """
    task = Task(TaskTypesEnum.classification)
    basis_models = get_operations_for_task(
        task=task, mode='data_operation', tags=["basis"])
    extractor = get_operations_for_task(
        task=task, mode='data_operation', tags=["extractor"])
    other = get_operations_for_task(
        task=task, forbidden_tags=["basis", "extractor"])

    for idx, node in enumerate(pipeline.nodes):
        # Operation name in the current node
        current_operation = node.operation.operation_type
        parent_nodes = node.nodes_from
        if parent_nodes:
            if current_operation in basis_models and parent_nodes[0].name != 'channel_filtration':
                raise ValueError(
                    f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination')
            # There are several parents for current node or at least 1
            for parent in parent_nodes:
                parent.operation.operation_type
                if current_operation in basis_models and \
                        pipeline.nodes[idx + 1].operation.operation_type not in extractor:
                    raise ValueError(
                        f'{ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination. '
                        f'Basis output should contain feature transformation')

        else:
            continue
    return True


def has_no_lagged_conflicts_in_ts_pipeline(pipeline: Pipeline):
    """ Function checks the correctness of connection between nodes """
    task = Task(TaskTypesEnum.ts_forecasting)
    non_lagged_models = get_operations_for_task(
        task=task, mode='model', tags=["non_lagged"])

    for idx, node in enumerate(pipeline.nodes):
        # Operation name in the current node
        current_operation = node.operation.operation_type
        parent_nodes = node.nodes_from
        if len(parent_nodes) == 0:
            return True
        for parent in parent_nodes:
            is_lagged = parent.name == 'lagged'
            check_condition = all(
                [current_operation in non_lagged_models, is_lagged])
            if check_condition:
                return False
    return True


def _crossover_by_type(self, crossover_type: CrossoverTypesEnum) -> None:
    IndustrialCrossover()
    return None


def reproduce_controlled_industrial(self,
                                    population: PopulationT,
                                    evaluator: EvaluationOperator,
                                    pop_size: Optional[int] = None,
                                    ) -> PopulationT:
    """Reproduces and evaluates population (select, crossover, mutate).
    Doesn't implement any additional checks on population.
    """
    # If operators can return unchanged individuals from previous population
    # (e.g. both Mutation & Crossover are not applied with some probability)
    # then there's a probability that duplicate individuals can appear

    selected_individuals = self.selection(population, pop_size)
    new_population = []  # industrial don use crossover
    if len(selected_individuals) < pop_size:
        for ind_to_reproduce in range(pop_size):
            try:
                random_ind = np.random.choice(selected_individuals)
                new_ind = self.mutation(random_ind)
                if isinstance(new_ind, Individual):
                    new_population.append(new_ind)
            except Exception:
                pass
    else:
        new_population = self.mutation(selected_individuals)
    new_population = ensure_wrapped_in_sequence(new_population)
    new_population = evaluator(new_population)
    return new_population


def reproduce_industrial(self,
                         population: PopulationT,
                         evaluator: EvaluationOperator
                         ) -> PopulationT:
    """Reproduces and evaluates population (select, crossover, mutate).
    Implements additional checks on population to ensure that population size
    follows required population size.
    """
    collected_next_population = {}
    population_size_to_achieve = round(
        self.parameters.pop_size * self.parameters.required_valid_ratio)
    MIN_POP_SIZE = 5
    self.stop_condition = False
    for i in range(3):
        # Estimate how many individuals we need to complete new population
        # based on average success rate of valid results
        residual_size = self.parameters.pop_size - \
            len(collected_next_population)
        residual_size = max(MIN_POP_SIZE, int(
            residual_size / self.mean_success_rate))
        # residual_size = min(len(population), residual_size)

        # Reproduce the required number of individuals that equals residual size
        partial_next_population = self.reproduce_uncontrolled(
            population, evaluator, residual_size)
        if partial_next_population is None:
            # timeout condition
            self.stop_condition = True
            return population
        # Avoid duplicate individuals that can come unchanged from previous population
        collected_next_population.update(
            {ind.uid: ind for ind in partial_next_population})

        # Keep running average of transform success rate (if sample is big enough)
        if len(partial_next_population) >= MIN_POP_SIZE:
            valid_ratio = len(partial_next_population) / residual_size
            self._success_rate_window = np.roll(
                self._success_rate_window, shift=1)
            self._success_rate_window[0] = valid_ratio

        # Successful return: got enough individuals
        if len(collected_next_population) >= population_size_to_achieve:
            self._log.info(f'Reproduction achieved pop size {len(collected_next_population)}'
                           f' using {i + 1} attempt(s) with success rate {self.mean_success_rate:.3f}')
            return list(collected_next_population.values())[:self.parameters.pop_size]
    else:
        return list(collected_next_population.values())
