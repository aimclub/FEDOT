from datetime import datetime
from lib2to3.pytree import type_repr
from re import I
from secrets import choice
import sys
from typing import Optional, Union, List

from numpy import append
parentdir = 'C:\\Users\\Worker1\\Documents\\FEDOT'
sys.path.insert(0, parentdir)

import time


from fedot.core.dag.graph import Graph
from joblib import PrintTime

from copy import deepcopy
import itertools
from fedot.core.dag.graph_node import GraphNode
 
import pandas as pd
import random
from functools import partial
from sklearn import preprocessing
import seaborn as sns
 
import bamt.Preprocessors as pp
from bamt.Builders import StructureBuilder
import bamt.Networks as Nets
 
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log, Log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score, BicScore, BDeuScore
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from fpdf import FPDF
from math import log10, ceil
from datetime import timedelta
from random import randint
from pomegranate import *
import networkx as nx

class CustomGraphModel(Graph):

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1



class CustomGraphNode(OptNode):
    def __str__(self):
        return self.content["name"]
 

def custom_metric(graph: CustomGraphModel, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    data_values=data.values
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        struct.append([l1, l2])
    
    new_struct=[ [] for _ in range(len(vertices))]
    for pair in struct:
        i=dir_of_vertices[pair[1]]
        j=dir_of_vertices[pair[0]]
        new_struct[i].append(j)
    
    new_struct=tuple(map(lambda x: tuple(x), new_struct))
    model = BayesianNetwork.from_structure(data_values, new_struct)
    model.fit(data_values)
    model.bake()
    L=model.log_probability(data_values)
    LL=L.sum()
    # Dim = (sum([len(new_struct[i]) for i in range(len(new_struct))]))/len(new_struct)
    # Dim = (sum([len(new_struct[i]) for i in range(len(new_struct))]))
    # Dim = len(graph_nx.edges())
    Dim = 0
    for i in nodes:
        unique = unique_values[i]
        for j in new_struct[dir_of_vertices[i]]:
            unique = unique * unique_values[dir_of_vertices_rev[j]]
        Dim += unique
    score = LL - log10(len(data)/2)*Dim
    return [-score]

def opt_graph_to_bamt(graph: CustomGraphModel):
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        if 'Node' in l1:
            l1 = l1.split('_')[1]
        if 'Node' in l2:
            l2 = l2.split('_')[1]
        struct.append((l1, l2))
    return struct 
 

def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
 
 
def _has_disc_parents(graph):
    graph, labels = graph_structure_as_nx_graph(graph)
    for pair in graph.edges():
        if (node_type[str(labels[pair[1]])] == 'disc') & (node_type[str(labels[pair[0]])] == 'cont'):
            raise ValueError(f'Discrete node has cont parent')
    return True

def _no_empty_graph(graph):
    graph, _ = graph_structure_as_nx_graph(graph)
    if len(graph.edges())==0:
        raise ValueError(f'Graph empty')
    return True
 
# меняем первый граф, второй такой же 
def custom_crossover_parents1(graph_first, graph_second, max_depth):

    num_cros = 1
    try:
        for _ in range(num_cros):
            new_graph_first=deepcopy(graph_first)

            dir_of_nodes={new_graph_first.nodes[i].content['name']:i for i in range(len(new_graph_first.nodes))}
            edges = graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node=random.choice(nodes_with_parent_or_child)
                parents=selected_node.nodes_from

                node_from_first_graph=new_graph_first.nodes[dir_of_nodes[selected_node.content['name']]]
            
                node_from_first_graph.nodes_from=[]
                if parents!=[] and parents!=None:
                    parents_in_first_graph=[new_graph_first.nodes[dir_of_nodes[i.content['name']]] for i in parents]
                    for i in range(len(parents_in_first_graph)):
                        node_from_first_graph.nodes_from.append(parents_in_first_graph[i])
    except Exception as ex:
        graph_first.show()
        print(ex)
 
    return new_graph_first, graph_second

# на слуйчай пустого графа
# меняем первый граф, меняем второй граф 
def custom_crossover_parents(graph_first, graph_second, max_depth):

    num_cros = 1
    try:
        for _ in range(num_cros):
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            dir_of_nodes1={new_graph_first.nodes[i].content['name']:i for i in range(len(new_graph_first.nodes))}
            dir_of_nodes2={new_graph_second.nodes[i].content['name']:i for i in range(len(new_graph_second.nodes))}

            edges = graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=random.choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=new_graph_first.nodes[dir_of_nodes1[selected_node2.content['name']]]
                parents1=selected_node1.nodes_from
            
                if parents2!=[] and parents2!=None:
                    selected_node1.nodes_from=[]
                    parents_in_first_graph=[new_graph_first.nodes[dir_of_nodes1[i.content['name']]] for i in parents2]
                    for i in range(len(parents_in_first_graph)):
                        selected_node1.nodes_from.append(parents_in_first_graph[i])
                if parents1!=[] and parents1!=None:
                    selected_node2.nodes_from=[]
                    parents_in_second_graph=[new_graph_second.nodes[dir_of_nodes2[i.content['name']]] for i in parents1]
                    for i in range(len(parents_in_second_graph)):
                        selected_node2.nodes_from.append(parents_in_second_graph[i])                

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

# не использовать, если начинаем с пустых графов
def custom_crossover_parents2(graph_first, graph_second, max_depth):

    num_cros = 1
    try:
        for _ in range(num_cros):
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            # dir_of_nodes1={new_graph_first.nodes[i].content['name']:i for i in range(len(new_graph_first.nodes))}
            # dir_of_nodes2={new_graph_second.nodes[i].content['name']:i for i in range(len(new_graph_second.nodes))}

            selected_node2=random.choice(new_graph_second.nodes)
            parents2=selected_node2.nodes_from

            selected_node1=new_graph_first.nodes[dir_of_nodes[selected_node2.content['name']]]
            parents1=selected_node1.nodes_from

            selected_node1.nodes_from=[]
            selected_node2.nodes_from=[]
            
            if parents2!=[] and parents2!=None:
                parents_in_first_graph=[new_graph_first.nodes[dir_of_nodes[i.content['name']]] for i in parents2]
                for i in range(len(parents_in_first_graph)):
                    selected_node1.nodes_from.append(parents_in_first_graph[i])
            if parents1!=[] and parents1!=None:
                parents_in_second_graph=[new_graph_second.nodes[dir_of_nodes[i.content['name']]] for i in parents1]
                for i in range(len(parents_in_second_graph)):
                    selected_node2.nodes_from.append(parents_in_second_graph[i])                

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second


def custom_crossover_parents3(graph_first, graph_second, max_depth):

    num_cros = 1
    try:
        for _ in range(num_cros):

            # def parent_exchange(graph, selected_node, parents):
            #     if parents!=[] and parents!=None:
            #         selected_node.nodes_from=[]
            #         parents_in_graph=[graph.nodes[dir_of_nodes[i.content['name']]] for i in parents]
            #         for i in range(len(parents_in_graph)):
            #             selected_node.nodes_from.append(parents_in_graph[i])
            #     else:
            #         selected_node.nodes_from=[]
            #     return graph

            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)


            # selected_node2=random.choice(new_graph_second.nodes)
            # parents2=selected_node2.nodes_from

            edges = graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:

                selected_node2=random.choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=new_graph_first.nodes[dir_of_nodes[selected_node2.content['name']]]
                parents1=selected_node1.nodes_from
                
                # new_graph_first = parent_exchange(new_graph_first, selected_node1, parents2)
                # new_graph_second = parent_exchange(new_graph_second, selected_node2, parents1)

                # parents2_1=parents2[0].nodes_from
                # parent_exchange(new_graph_first, selected_node, parents)

            # new_graph_first.update_subtree(selected_node1, selected_node2)
            # new_graph_first.show()
                
            selected_node1.nodes_from=[]    
            if parents2!=[] and parents2!=None:
                parents_in_first_graph=[new_graph_first.nodes[dir_of_nodes[i.content['name']]] for i in parents2]
                for i in range(len(parents_in_first_graph)):
                    p_i = parents_in_first_graph[i]
                    selected_node1.nodes_from.append(p_i)
                    next_par = new_graph_second.nodes[dir_of_nodes[p_i.content['name']]].nodes_from
                    if next_par==[] or next_par==None:
                        continue
                    else:
                        p_i.nodes_from=[]                       
                        next_par_first = [new_graph_first.nodes[dir_of_nodes[i.content['name']]] for i in next_par]
                        for j in range(len(next_par_first)):
                            p_i.nodes_from.append(next_par_first[j])

            selected_node2.nodes_from=[]    
            if parents1!=[] and parents1!=None:
                parents_in_second_graph=[new_graph_second.nodes[dir_of_nodes[i.content['name']]] for i in parents1]
                for i in range(len(parents_in_second_graph)):
                    p_i = parents_in_second_graph[i]
                    selected_node2.nodes_from.append(p_i)
                    next_par = new_graph_first.nodes[dir_of_nodes[p_i.content['name']]].nodes_from
                    if next_par==[] or next_par==None:
                        continue                    
                    else:
                        p_i.nodes_from=[]                            
                        next_par_second = [new_graph_second.nodes[dir_of_nodes[i.content['name']]] for i in next_par]
                        for j in range(len(next_par_second)):
                            p_i.nodes_from.append(next_par_second[j])

            # if parents1!=[] and parents1!=None:
            #     selected_node2.nodes_from=[]
            #     parents_in_second_graph=[new_graph_second.nodes[dir_of_nodes[i.content['name']]] for i in parents1]
            #     for i in range(len(parents_in_second_graph)):
            #         selected_node2.nodes_from.append(parents_in_second_graph[i])                

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_parents4(graph_first: OptGraph, graph_second: OptGraph, max_depth):

    num_cros = 1
    try:
        for _ in range(num_cros):
            def find_node(graph: OptGraph, node):
                return graph.nodes[dir_of_nodes[node.content['name']]]
                 


            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)
            # new_graph_first.show()
            # new_graph_second.show()

            edges_1 = new_graph_first.operator.get_all_edges()
            edges_2 = new_graph_second.operator.get_all_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = random.sample(edges_1, count)
            choice_edges_2 = random.sample(edges_2, count)
            
            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)  
            

            new_edges_2 = [[find_node(new_graph_second, i[0]), find_node(new_graph_second, i[1])] for i in choice_edges_1]
            new_edges_1 = [[find_node(new_graph_first, i[0]), find_node(new_graph_first, i[1])] for i in choice_edges_2] 
            for pair in new_edges_1:
                new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                new_graph_second.operator.connect_nodes(pair[0], pair[1])                        
            # flatten_edges = list(itertools.chain(*edges))
            # nodes_with_parent_or_child=list(set(flatten_edges))

            # new_graph_first.show()
            # new_graph_second.show()       

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_mutation_add(graph: OptGraph, **kwargs):
    num_mut = 10
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
    num_mut = 10
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
    num_mut = 10
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



def run_example():

    global vertices
    vertices = list(data.columns)
    global dir_of_vertices
    dir_of_vertices={vertices[i]:i for i in range(len(vertices))}    
    global dir_of_vertices_rev
    dir_of_vertices_rev={i:vertices[i] for i in range(len(vertices))}    
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    global unique_values
    unique_values = {vertices[i]:len(pd.unique(discretized_data[vertices[i]])) for i in range(len(vertices))}
    global node_type
    node_type = p.info['types'] 
    types=list(node_type.values())

    if 'cont' in types and 'disc' in types:
        bn = Nets.HybridBN(has_logit=False, use_mixture=False)
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    elif 'disc' in types:
        bn = Nets.DiscreteBN()
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    elif 'cont' in types:
        bn = Nets.ContinuousBN()
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    bn.add_nodes(p.info)
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score))

    #rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]


    # custom_mutation_add(initial[0]).show()
    # [initial=custom_mutation_add(initial[0]) for _ in range(10)]
    



    
    #initial[0].show()
    # for node in initial[0].nodes: 
    #     parents = []
    #     for n in bn.nodes:
    #         if str(node) == str(n):
    #             parents = n.cont_parents + n.disc_parents
    #             break
    #     for n2 in initial[0].nodes:
    #         if str(n2) in parents:
    #             node.nodes_from.append(n2)
                


                # OF=round(custom_metric(initial[0], method=met, data=discretized_data)[0],2)
                # print(met, '=', OF)
                # initial[0].show()
    
    OF_init=(round(custom_metric(initial[0], data=discretized_data)[0],2))
    initial[0].show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V_init' + str(j) + '.png'))
    global dir_of_nodes
    dir_of_nodes={initial[0].nodes[i].content['name']:i for i in range(len(initial[0].nodes))} 

    # задать рандомную начальную популяцию. Последнего сделает по _create_randomized_pop_from_inital_graph. 
    # Без этого не выйдет из цикла
    

    # pop_size=100

    # list_of_time=[]
    # start_time = time.perf_counter()

    init=initial[0]
    initial=[]
    for i in range(0, pop_size-1):
        rand = randint(1, 2*len(vertices))
        g=deepcopy(init)
        for _ in range(rand):
            g=deepcopy(custom_mutation_add(g))
        initial.append(g)
    
    # elapsed_time = time.perf_counter() - start_time 
    # list_of_time.append(elapsed_time)
    # print(list_of_time)
    # print(1)
    
    # list_of_time=[]
    # start_time = time.perf_counter()



    # number_nodes = len(vertices)
    # DAG_list=[]
    # init=initial[0]
    # while len(DAG_list)<pop_size-1:
    #     is_all = True
    #     DAG = []
    #     G=nx.gnp_random_graph(number_nodes,0.5,directed=True)
    #     DAG = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    #     DAG_list.append(DAG)

    # li=list(map(lambda x: x.edges(), DAG_list))

    # initial = []
    # for i in range(pop_size-1):
    #     init_r = deepcopy(init)
    #     for pair in li[i]:
    #         init_r.nodes[pair[0]].nodes_from.append(init_r.nodes[pair[1]])
    #     initial.append(init_r)
    


    # elapsed_time = time.perf_counter() - start_time 
    # list_of_time.append(elapsed_time)
    # print(list_of_time)


    # li_new = []
    # for (pair1, pair2) in li[0]:
    #     v1 = dir_of_vertices_rev[pair1]
    #     v2 = dir_of_vertices_rev[pair2]
    #     li_new.append([v1, v2])
    # print(li_new)

    # for node in initial[0].nodes: 
    #     parents = []
    #     for pair in li_new:
    #         if str(node) == pair[0]:
    #             parents.append(pair[1])
    #     for n2 in initial[0].nodes:
    #         if str(n2) in parents:
    #             node.nodes_from.append(n2)
        

    requirements = PipelineComposerRequirements(
        primary=vertices,
        secondary=vertices, max_arity=100,
        max_depth=100, pop_size=pop_size, num_of_generations=10000,
        crossover_prob=0.8, mutation_prob=0.9, timeout=timedelta(minutes=time_m))
 
    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        selection_types=[SelectionTypesEnum.tournament],
        mutation_types=[custom_mutation_add, custom_mutation_delete, custom_mutation_reverse],
        crossover_types=[custom_crossover_parents4],
        # crossover_types=[custom_crossover_parents1, custom_crossover_parents],
        regularization_type=RegularizationTypesEnum.none,
        # если улучшение не происходит в течении ... поколений -> выход
        stopping_after_n_generation=10000
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='Bayesian', verbose_level=1))


    optimized_graph = optimiser.optimise(partial(custom_metric, data=discretized_data))
    optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
    structure = opt_graph_to_bamt(optimized_network)
    optimized_graph.show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(j) + '.png')) 
    OF=round(custom_metric(optimized_network, data=discretized_data)[0],2)
  

    #final_bn = Nets.DiscreteBN()
    # if 'cont' in types and 'disc' in types:
    #     final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    # elif 'disc' in types:
    #     final_bn = Nets.DiscreteBN()
    # elif 'cont' in types:
    #     final_bn = Nets.ContinuousBN()

    # final_bn.add_nodes(p.info)
    # structure = opt_graph_to_bamt(optimized_network)
    # #print(structure)
    # final_bn.set_structure(edges=structure)
    # final_bn.get_info()
    # final_bn.fit_parameters(data)

    # prediction = dict()

    # for c in vertices:
    #     test = data.drop(columns=[c])
    #     pred = final_bn.predict(test, 5)
    #     prediction.update(pred)
    
    # result = dict()
    # for key, value in prediction.items():
    #     if node_type[key]=="disc":
    #         res=round(accuracy_score(data[key], value),2)
    #     elif node_type[key]=="cont":
    #         res=round(mean_squared_error(data[key], value, squared=False),2)
    #     result[key]=res


    def child_dict(net: list):
        res_dict = dict()
        for e0, e1 in net:
            if e1 in res_dict:
                res_dict[e1].append(e0)
            else:
                res_dict[e1] = [e0]
        return res_dict

    def precision_recall(pred_net: list, true_net: list, decimal = 2):
        pred_dict = child_dict(pred_net)
        true_dict = child_dict(true_net)
        corr_undir = 0
        corr_dir = 0
        for e0, e1 in pred_net:
            flag = True
            if e1 in true_dict:
                if e0 in true_dict[e1]:
                    corr_undir += 1
                    corr_dir += 1
                    flag = False
            if (e0 in true_dict) and flag:
                if e1 in true_dict[e0]:
                    corr_undir += 1
        pred_len = len(pred_net)
        true_len = len(true_net)
        shd = pred_len + true_len - corr_undir - corr_dir
        return {'AP': round(corr_undir/pred_len, decimal), 
        'AR': round(corr_undir/true_len, decimal), 
        'AHP': round(corr_dir/pred_len, decimal), 
        'AHR': round(corr_dir/true_len, decimal), 
        'SHD': shd}

    
    SHD=precision_recall(structure, true_net)['SHD']

    # structure_init = opt_graph_to_bamt(initial[0])
    # SHD_init=[]
    # SHD_init=precision_recall(structure_init, true_net)['SHD']

    pdf.add_page()
    pdf.set_font("Arial", size = 14)
    
    pdf.cell(150, 5, txt = str(OF),
            ln = 1, align = 'C')
    pdf.cell(150, 5, txt = "pop_size = " + str(requirements.pop_size),
            ln = 1)
    pdf.cell(150, 5, txt = "mutation_prob = " + str(requirements.mutation_prob),
            ln = 1)
    pdf.cell(150, 5, txt = "crossover_prob = " + str(requirements.crossover_prob),
            ln = 1)            
    pdf.cell(150, 5, txt = "genetic_scheme_type = " + str(optimiser_parameters.genetic_scheme_type),
            ln = 1)
    pdf.cell(150, 5, txt = "selection_types = " + str(optimiser_parameters.selection_types),
            ln = 1)
    pdf.multi_cell(180, 5, txt = "mutation_types = " + str(optimiser_parameters.mutation_types))
    pdf.multi_cell(180, 5, txt = "crossover_types = " + str(optimiser_parameters.crossover_types))
    pdf.cell(150, 5, txt = "stopping_after_n_generation = " + str(optimiser_parameters.stopping_after_n_generation),
            ln = 1)
    pdf.cell(150, 5, txt = "actual_generation_num = " + str(optimiser.generation_num),
            ln = 1)     
    pdf.cell(150, 5, txt = "timout = " + str(j),
            ln = 1)          
    pdf.image('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(j) + '.png',w=165, h=165)
    pdf.multi_cell(180, 5, txt = str(structure))
    pdf.multi_cell(180, 5, txt = 'SHD = '+str(SHD))

    # pdf.add_page()
    # pdf.set_font("Arial", size = 14)
    # pdf.cell(150, 5, txt = 'BAMT',
    #         ln = 1, align = 'C')
    # pdf.cell(150, 5, txt = str(OF_init),
    #         ln = 1, align = 'C')
    # pdf.image('C:/Users/Worker1/Documents/FEDOT/examples/V_init' + str(j) + '.png',w=165, h=165)
    # pdf.multi_cell(180, 5, txt = str(structure_init))
    # pdf.multi_cell(180, 5, txt = 'SHD = '+str(SHD_init))

    # for heading, row in result.items():
    #     pdf.cell(40, 6, heading, 1)
    #     pdf.cell(40, 6, str(row), 1)
    #     pdf.ln()
    

if __name__ == '__main__':
    # data = pd.read_csv(r'examples/data/asia.csv')
    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # nodes = list(data.columns)
    #files = ['asia', 'sachs', 'magic-niab', 'ecoli70', 'child']
    pop_size=100
    files = ['child']
    for file in files:
        data = pd.read_csv('examples/data/'+file+'.csv')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        # print(data.isna().sum())
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)
        
        with open('examples/data/'+file+'.txt') as f:
            lines = f.readlines()
        true_net = []
        for l in lines:
            e0 = l.split()[0]
            e1 = l.split()[1].split('\n')[0]
            true_net.append((e0, e1))

        # pdf = FPDF()  
        
        # for j in [1, 2]+list(range(5, 40, 5)):
        
            # pdf = FPDF()
            # time_m=j
            # structure = run_example()
            # pdf.output("f_Experiment_empty_"+file + str(j) + ".pdf")
        try:
            for j in [1, 2]+list(range(5, 20, 5)):
                pdf = FPDF()
                time_m=j
                structure = run_example()
                pdf.output("123Experiment_random_init_"+file+str(j)+".pdf")
        except Exception as ex:
            print(ex)
            pdf.output(str(time.time())+".pdf")

    # for i, met in zip(range(1,4), ['K2','BDeu','Bic']):
    #     pdf = FPDF()
    #     for j in range(0):
    #         structure = run_example()
    #     pdf.output("New_custom_metric_asia.pdf")