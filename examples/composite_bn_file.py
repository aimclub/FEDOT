
import sys
from typing import Optional, Union, List


parentdir = 'C:\\Users\\anaxa\\Documents\\Projects\\\CompositeBayesianNetworks\\FEDOT'
sys.path.insert(0, parentdir)
from fedot.core.dag.graph import Graph
from copy import deepcopy
import pandas as pd
import random
from sklearn import preprocessing
import bamt.Preprocessors as pp
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer, GPGraphOptimizerParameters
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score
from math import ceil
from examples.composite_model import CompositeModel
from examples.composite_node import CompositeNode



# задаем метрику
def custom_metric(graph: CompositeModel, data: pd.DataFrame):
    score = 0
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        struct.append([l1, l2])
    
    bn_model = BayesianNetwork(struct)
    bn_model.add_nodes_from(data.columns)    
    
    score = K2Score(data).score(bn_model)
    return [-score]

# задаем кроссовер (обмен ребрами)
def custom_crossover_exchange_edges(graph_first: OptGraph, graph_second: OptGraph, max_depth):
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges_1 = new_graph_first.operator.get_edges()
            edges_2 = new_graph_second.operator.get_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = random.sample(edges_1, count)
            choice_edges_2 = random.sample(edges_2, count)
            
            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)  
            
            old_edges1 = new_graph_first.operator.get_edges()
            old_edges2 = new_graph_second.operator.get_edges()

            new_edges_2 = [[find_node(new_graph_second, i[0]), find_node(new_graph_second, i[1])] for i in choice_edges_1]
            new_edges_1 = [[find_node(new_graph_first, i[0]), find_node(new_graph_first, i[1])] for i in choice_edges_2] 
            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])                                             
            
            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break

        if old_edges1 == new_graph_first.operator.get_edges() and new_edges_1!=[] and new_edges_1!=None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_edges() and new_edges_2!=[] and new_edges_2!=None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second



# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 

def custom_mutation_delete(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
                break
    except Exception as ex:
        print(ex) 
    return graph


def custom_mutation_reverse(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)   
                break         
    except Exception as ex:
        print(ex)  
    return graph


# задаем правила на запрет дублирующих узлов
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
    

def run_example():

    data = pd.read_csv('examples/data/'+file+'.csv')   
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    vertices = list(data.columns)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, _ = p.apply(data)

    # словарь: {имя_узла: уникальный_номер_узла}
    global dir_of_nodes
    dir_of_nodes={data.columns[i]:i for i in range(len(data.columns))}     

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    

    # задаем для оптимизатора fitness-функцию
    objective = Objective(custom_metric) 
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    
    # инициализация начальной сети (пустая)
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                       content={'name': vertex}) for vertex in vertices])]


    requirements = PipelineComposerRequirements(
        primary=vertices,
        secondary=vertices, 
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation
        )

    optimiser_parameters = GPGraphOptimizerParameters(
        pop_size=pop_size,
        crossover_prob=crossover_probability, 
        mutation_prob=mutation_probability,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = [custom_mutation_add, custom_mutation_delete, custom_mutation_reverse],
        crossover_types = [custom_crossover_exchange_edges]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=initial,
        objective=objective)

    


    def reverse_edge(self, node_parent, node_child):
        self.disconnect_nodes(node_parent, node_child, False)
        self.connect_nodes(node_child, node_parent)

    def connect_nodes(self, parent, child):
        if child.descriptive_id not in [p.descriptive_id for p in parent.ordered_subnodes_hierarchy()]:
            try:
                if child.nodes_from==None:
                    child.nodes_from=[]
                child.nodes_from.append(parent)
            except Exception as ex:
                print(ex)
    
    GraphOperator.reverse_edge = reverse_edge
    GraphOperator.connect_nodes = connect_nodes

    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    # вывод полученного графа
    optimized_graph.show()


if __name__ == '__main__':

    # файл с исходными данными (должен лежать в 'examples/data/')
    file = 'asia'     
    # размер популяции
    pop_size = 20
    # количество поколений
    n_generation = 50
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9

    run_example() 