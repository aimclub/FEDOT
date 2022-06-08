from ast import excepthandler
from datetime import datetime
from lib2to3.pytree import type_repr
from re import I
from secrets import choice
import sys
from telnetlib import X3PAD, XASCII
from typing import Optional, Union, List
import bnlearn as bn
from numpy import append
from yaml import YAMLObject
parentdir = 'C:\\Users\\Worker1\\Documents\\FEDOT'
sys.path.insert(0, parentdir)

import time
from itertools import permutations

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
from pyitlib import discrete_random_variable as drv

 
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
 

def custom_metric_LL(graph: CustomGraphModel, data: pd.DataFrame):
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

def metric_for_structure_LL(struc, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    data_values=data.values
    struct = struc
    
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
    Dim = 0
    for i in nodes:
        unique = unique_values[i]
        for j in new_struct[dir_of_vertices[i]]:
            unique = unique * unique_values[dir_of_vertices_rev[j]]
        Dim += unique
    score = LL - log10(len(data)/2)*Dim
    return [-score]    


# метрика на основе заполнения пропусков
def custom_metric_pass(graph: CustomGraphModel, data: pd.DataFrame):

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    
   


    final_bn = []
    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        final_bn = Nets.DiscreteBN()
    elif 'cont' in p.info['types'].values():
        final_bn = Nets.ContinuousBN(use_mixture=False)
    
    final_bn.add_nodes(p.info)
    structure = opt_graph_to_bamt(graph)
    # print(structure)
  

    final_bn.set_structure(edges=structure)
    final_bn.fit_parameters(data)

    prediction = dict()
    for c in vertices:
        test = data.drop(columns=[c])
        pred = final_bn.predict(test,5)
        prediction.update(pred)    


    result = dict()
    for key, value in prediction.items():
        if node_type[key]=="disc":
            res=1 - round(accuracy_score(data[key], value),2)
        elif node_type[key]=="cont":
            res=(round(mean_squared_error(data[key], value, squared=False),2)) / (data[key].max() - data[key].min())
        result[key]=res
    score = sum(result.values())
    return [score]

def metric_for_structure_pass(struc, data: pd.DataFrame):

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    
   


    final_bn = []
    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        final_bn = Nets.DiscreteBN()
    elif 'cont' in p.info['types'].values():
        final_bn = Nets.ContinuousBN(use_mixture=False)
    
    final_bn.add_nodes(p.info)
    structure = struc
    # print(structure)
  

    final_bn.set_structure(edges=structure)
    final_bn.fit_parameters(data)

    prediction = dict()
    for c in vertices:
        test = data.drop(columns=[c])
        pred = final_bn.predict(test,5)
        prediction.update(pred)    


    result = dict()
    for key, value in prediction.items():
        if node_type[key]=="disc":
            res=1 - round(accuracy_score(data[key], value),2)
        elif node_type[key]=="cont":
            res=(round(mean_squared_error(data[key], value, squared=False),2)) / (data[key].max() - data[key].min())
        result[key]=res
    score = sum(result.values())
    return [score]



def custom_metric_mi(graph: CustomGraphModel, data: pd.DataFrame):
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
    
    
    def mi(struct, vertices):
        MI = 0
        for i in vertices:
            if i in list(itertools.chain(*struct)):
                arcs = [j for j in struct if j[0]==i or j[1]==i]
                for a in arcs:
                    MI += drv.information_mutual(discretized_data[a[0]].values, discretized_data[a[1]].values)
            else: 
                MI += drv.information_mutual(discretized_data[i].values, discretized_data[i].values)
        return(MI)

    score = mi(struct, nodes)
    return [score]

def metric_for_structure_mi(struc, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    data_values=data.values
    struct = struc

    new_struct=[ [] for _ in range(len(vertices))]
    for pair in struct:
        i=dir_of_vertices[pair[1]]
        j=dir_of_vertices[pair[0]]
        new_struct[i].append(j)
    
    new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
    def mi(struct, vertices):
        MI = 0
        for i in vertices:
            if i in list(itertools.chain(*struct)):
                arcs = [j for j in struct if j[0]==i or j[1]==i]
                for a in arcs:
                    MI += drv.information_mutual(discretized_data[a[0]].values, discretized_data[a[1]].values)
            else: 
                MI += drv.information_mutual(discretized_data[i].values, discretized_data[i].values)
        return(MI)

    score = mi(struct, nodes)
    
    return [score]    


def custom_metric_cmi(graph: CustomGraphModel, data: pd.DataFrame):
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
    
    def cmi(stru, vertices):
        CMI = 0
        for X in vertices:
            if X in [k[1] for k in stru]:
                parents = [j[0] for j in stru if j[1]==X]
                if len(parents)==1:
                    Y=parents[0]
                    CMI += drv.information_mutual(discretized_data[X].values, discretized_data[Y].values)
                else:
                    for Y in parents:
                        par = deepcopy(parents)
                        par.remove(Y)
                        df_Z_int=deepcopy(discretized_data[par])
                        df_Z_str = df_Z_int.astype(str)
                        df_Z = df_Z_str.agg('-'.join, axis=1)
                        enc = preprocessing.LabelEncoder()
                        Z = enc.fit_transform(df_Z)
                        df = deepcopy(discretized_data[[X,Y]])
                        df.insert(2, 'Z', Z)
                        CMI += drv.information_mutual_conditional(df[X].values, df[Y].values, df['Z'].values)
            else: 
                CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
        return(CMI)   


    score = cmi(struct, nodes)
    return [score]

def metric_for_structure_cmi(struc, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    data_values=data.values
    struct = struc

    new_struct=[ [] for _ in range(len(vertices))]
    for pair in struct:
        i=dir_of_vertices[pair[1]]
        j=dir_of_vertices[pair[0]]
        new_struct[i].append(j)
    
    new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
    def cmi(stru, vertices):
        CMI = 0
        for X in vertices:
            if X in [k[1] for k in stru]:
                parents = [j[0] for j in stru if j[1]==X]
                if len(parents)==1:
                    Y=parents[0]
                    CMI += drv.information_mutual(discretized_data[X].values, discretized_data[Y].values)
                else:
                    for Y in parents:
                        par = deepcopy(parents)
                        par.remove(Y)
                        df_Z_int=deepcopy(discretized_data[par])
                        df_Z_str = df_Z_int.astype(str)
                        df_Z = df_Z_str.agg('-'.join, axis=1)
                        enc = preprocessing.LabelEncoder()
                        Z = enc.fit_transform(df_Z)
                        df = deepcopy(discretized_data[[X,Y]])
                        df.insert(2, 'Z', Z)
                        CMI += drv.information_mutual_conditional(df[X].values, df[Y].values, df['Z'].values)
            else: 
                CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
        return(CMI)  


    score = cmi(struct, nodes)
    
    return [score]  

def custom_metric_cmi_new(graph: CustomGraphModel, data: pd.DataFrame):
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
    
    def cmi(stru, vertices):
        CMI = 0
        for X in vertices:
            if X in [k[1] for k in stru]:
                parents = [j[0] for j in stru if j[1]==X]
                for Y in parents:
                    CMI += drv.information_mutual_conditional(discretized_data[X].values, discretized_data[X].values, discretized_data[Y].values)
            else: 
                CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
        return(CMI)   


    score = cmi(struct, nodes)
    return [score]

def metric_for_structure_cmi_new(struc, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    data_values=data.values
    struct = struc

    new_struct=[ [] for _ in range(len(vertices))]
    for pair in struct:
        i=dir_of_vertices[pair[1]]
        j=dir_of_vertices[pair[0]]
        new_struct[i].append(j)
    
    new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
    def cmi(stru, vertices):
        CMI = 0
        for X in vertices:
            if X in [k[1] for k in stru]:
                parents = [j[0] for j in stru if j[1]==X]
                for Y in parents:
                    CMI += drv.information_mutual_conditional(discretized_data[X].values, discretized_data[X].values, discretized_data[Y].values)
            else: 
                CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
        return(CMI)   


    score = cmi(struct, nodes)
    
    return [score]  

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
    # print('Родители')
    # graph_first.show()
    # graph_second.show()

    num_cros = 100
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
            

            old_edges1 = new_graph_first.operator.get_all_edges()
            old_edges2 = new_graph_second.operator.get_all_edges()

            new_edges_2 = [[find_node(new_graph_second, i[0]), find_node(new_graph_second, i[1])] for i in choice_edges_1]
            new_edges_1 = [[find_node(new_graph_first, i[0]), find_node(new_graph_first, i[1])] for i in choice_edges_2] 
            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])                                             
            
            if old_edges1 != new_graph_first.operator.get_all_edges() or old_edges2 != new_graph_second.operator.get_all_edges():
                break

            
            # flatten_edges = list(itertools.chain(*edges))
            # nodes_with_parent_or_child=list(set(flatten_edges))

            # new_graph_first.show()
            # new_graph_second.show()
            # print('Дети') 
            # new_graph_first.show()      
            # new_graph_second.show()

        if old_edges1 == new_graph_first.operator.get_all_edges():
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_all_edges():
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_parents5(graph_first: OptGraph, graph_second: OptGraph, max_depth):
    num_cros = 1
    # print(graph_first.operator.get_all_edges())
    # print(graph_second.operator.get_all_edges())

    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]
    try:
        for _ in range(num_cros):
            new_graph_first = deepcopy(graph_first)
            new_graph_second = deepcopy(graph_second)      
            
            rid = random.choice(range(len(new_graph_first.nodes)))
            node_f = new_graph_first.nodes[rid] 
            node_s = find_node(new_graph_second, node_f)
            par_f = node_f.nodes_from
            par_s = node_s.nodes_from
            # print(node_f)
            # print(par_f)
            # print(par_s)


            node_f.nodes_from = []
            node_s.nodes_from = []

            if par_s!=[] and par_s!=None:
                par_f_new = [find_node(new_graph_first, i) for i in par_s]  
                for i in par_f_new:
                    node_f.nodes_from.append(i)

            if par_f!=[] and par_f!=None:
                par_s_new = [find_node(new_graph_second, i) for i in par_f] 
                for i in par_s_new:
                    node_s.nodes_from.append(i)            


    except Exception as ex:
        print(ex)
    # print(new_graph_first.operator.get_all_edges())
    # print(new_graph_second.operator.get_all_edges())
    return new_graph_first, new_graph_second

def custom_mutation_add(graph: OptGraph, **kwargs):
    num_mut = 100
    m=0
    try:
        for _ in range(num_mut):
            m+=1
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
    # if m==20:
    #     print('10 раз попытался сделать добавление ребра')
    return graph
 

def custom_mutation_delete(graph: OptGraph, **kwargs):
    num_mut = 100
    m=0
    try:
        for _ in range(num_mut):
            m+=1
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
                break
    except Exception as ex:
        print(ex)
    # if m==20:
    #     print('10 раз попытался сделать удаление ребра')    
    return graph


def custom_mutation_reverse(graph: OptGraph, **kwargs):
    num_mut = 100
    m=0
    try:
        for _ in range(num_mut):
            m+=1
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)   
                break         
    except Exception as ex:
        print(ex)
    # if m==20:
    #     print('10 раз попытался сделать реверс ребра')     
    return graph

def check_iequv(ind1, ind2):

        ## skeletons and immoralities
        (ske1, immor1) = get_skeleton_immor(ind1)
        (ske2, immor2) = get_skeleton_immor(ind2)

        ## comparison. 
        if len(ske1) != len(ske2) or len(immor1) != len(immor2):
            return False

        ## Note that the edges are undirected so we need to check both ordering
        for (n1, n2) in immor1:
            if (n1, n2) not in immor2 and (n2, n1) not in immor2:
                return False
        for (n1, n2) in ske1:
            if (n1, n2) not in ske2 and (n2, n1) not in ske2:
                return False
        return True


def get_skeleton_immor(ind):
    ## skeleton: a list of edges (undirected)
    skeleton = get_skeleton(ind)
    ## find immoralities
    immoral = set()
    for n in ind.nodes:
        if n.nodes_from != None and len(n.nodes_from) > 1:
            perm = list(permutations(n.nodes_from, 2))
            for (per1, per2) in perm:
                p1 = per1.content["name"]
                p2 = per2.content["name"]
                if ((p1, p2) not in skeleton and (p2, p1) not in skeleton 
                    and (p1, p2) not in immoral and (p2, p1) not in immoral):
                    immoral.add((p1, p2))

    return (skeleton, immoral)    

def get_skeleton(ind):
    skeleton = set()
    edges = ind.operator.get_all_edges()
    for e in edges:
        skeleton.add((e[0].content["name"], e[1].content["name"]))
        skeleton.add((e[1].content["name"], e[0].content["name"]))
    return skeleton


def run_example():

    global vertices
    vertices = list(data.columns)
    global dir_of_vertices
    dir_of_vertices={vertices[i]:i for i in range(len(vertices))}    
    global dir_of_vertices_rev
    dir_of_vertices_rev={i:vertices[i] for i in range(len(vertices))}    
    
    # data = pd.read_csv('examples/data/'+'asia'+'.csv')
    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    #     # print(data.isna().sum())
    # data.dropna(inplace=True)
    # data.reset_index(inplace=True, drop=True)
    

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # global p
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    global discretized_data
    discretized_data, est = p.apply(data)

    for i in discretized_data.columns:
        if discretized_data[i].dtype.name == 'int64':
            discretized_data = discretized_data.astype({i:'int32'})  
    
    global unique_values
    unique_values = {vertices[i]:len(pd.unique(discretized_data[vertices[i]])) for i in range(len(vertices))}
    global node_type
    node_type = p.info['types'] 
    global types
    types=list(node_type.values())

    if 'cont' in types and ('disc' in types or 'disc_num' in types):
        bn = Nets.HybridBN(has_logit=False, use_mixture=False)
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    elif 'disc' in types or 'disc_num' in types:
        bn = Nets.DiscreteBN()
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    elif 'cont' in types:
        bn = Nets.ContinuousBN(use_mixture=False)
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    # struct = [('asia', 'xray'), ('tub', 'bronc'), ('smoke', 'asia'), ('lung', 'either'), ('lung', 'xray'), ('bronc', 'dysp'), ('either', 'smoke'), ('xray', 'tub'), ('xray', 'dysp')]

    # bn = Nets.DiscreteBN()
    bn.add_nodes(p.info)
    # bn.set_structure(edges=struct)
    # bn.fit_parameters(data)
    # for c in data.columns:
    #     print(c)
    #     test = data.drop(columns=[c])
    #     bn.predict(test)
        
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score))

    #rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]

    init=deepcopy(initial[0])
    # custom_mutation_add(initial[0]).show()
    # [initial=custom_mutation_add(initial[0]) for _ in range(10)]
    # print('init score',custom_metric_LL(init,data=discretized_data))
    



    
    #initial[0].show()
    for node in initial[0].nodes: 
        parents = []
        for n in bn.nodes:
            if str(node) == str(n):
                parents = n.cont_parents + n.disc_parents
                break
        for n2 in initial[0].nodes:
            if str(n2) in parents:
                node.nodes_from.append(n2)
                
    BAMT_network = deepcopy(initial[0])
    structure_BAMT = opt_graph_to_bamt(BAMT_network)
    Score_BAMT = round(metric(BAMT_network, data=discretized_data)[0],3)

                # OF=round(metric(initial[0], method=met, data=discretized_data)[0],2)
                # print(met, '=', OF)
                # initial[0].show()
    
    # OF_init=(round(metric(initial[0], data=data)[0],2))
    # print(OF_init)
    # initial[0].show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V_init' + str(j) + '.png'))
    
    global dir_of_nodes
    dir_of_nodes={initial[0].nodes[i].content['name']:i for i in range(len(initial[0].nodes))} 

    # задать рандомную начальную популяцию. Последнего сделает по _create_randomized_pop_from_inital_graph. 
    # Без этого не выйдет из цикла
    

    # pop_size=100
    # list_of_time=[]
    # start_time = time.perf_counter()
    print('Создание начальной популяции')
    # генерация рандомных индивидов, соответствующих правилам
    initial=[]
    if len(vertices)<5:
    # без учета эквивалентности в популяции
        for i in range(0, pop_size):
            rand = randint(1, 2*len(vertices))
            g=deepcopy(init)
            for _ in range(rand):
                g=deepcopy(custom_mutation_add(g))
            initial.append(g)
    else:
    # с учетом эквивалентности в популяции (нет БС одного класса эквивалентности в популяции)
        for i in range(0, pop_size):
            rand = randint(1, 2*len(vertices))
            fl1 = False
            while not fl1:
                try:
                    fl1 = False
                    g=deepcopy(init)
                    for _ in range(rand):
                        g=deepcopy(custom_mutation_add(g))
                    mylist = []
                    for rule_func in rules:
                        mylist.append(rule_func(g))
                    fl1=all(mylist)
                    if fl1 and len(initial) != 0:
                        for i in initial:
                            if check_iequv(g, i):
                                fl1 = False
                                break
                except:
                    pass
            initial.append(g)




    print('Конец создания начальной популяции')

 
        # for (ind1, ind2) in permutations(self.population, 2):
        #     res = self.check_iequv(ind1, ind2)
    # elapsed_time = time.perf_counter() - start_time 
    # list_of_time.append(elapsed_time)
    # print(list_of_time)
    
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
        return {
        # 'AP': round(corr_undir/pred_len, decimal), 
        # 'AR': round(corr_undir/true_len, decimal), 
        # 'AHP': round(corr_dir/pred_len, decimal), 
        # 'AHR': round(corr_dir/true_len, decimal), 
        'SHD': shd}

    

    print('Вход')

    if micro:
        start_time = time.perf_counter()
        requirements = PipelineComposerRequirements(
            primary=vertices,
            secondary=vertices, max_arity=100,
            max_depth=100, pop_size=pop_size, num_of_generations=n_generation,
            crossover_prob=crossover_probability, mutation_prob=mutation_probability
            )
    
        optimiser_parameters = GPGraphOptimiserParameters(
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            selection_types=[SelectionTypesEnum.tournament],
            mutation_types=mutation_fun,
            crossover_types=crossover_fun,
            regularization_type=RegularizationTypesEnum.none,
            stopping_after_n_generation=10
        )

        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
            rules_for_constraint=rules)

        optimiser_parameters.custom = discretized_data
        if nich:
            optimiser_parameters.niching = []
        optimiser = EvoGraphOptimiser(
            graph_generation_params=graph_generation_params,
            metrics=[],
            parameters=optimiser_parameters,
            requirements=requirements, initial_graph=initial,
            log=default_log(logger_name='Bayesian', verbose_level=1))


# минуты
        elapsed_time =(time.perf_counter() - start_time)/60
        l_n = 0
        last = 0
        it = 0
        while l_n <=10 and elapsed_time < j and it < max_numb_nich:
            it+=1
            res_opt = optimiser.optimise(partial(metric, data=discretized_data))
            score = round(metric(res_opt, data=discretized_data)[0],2)

            

            if nich:
                optimiser_parameters.niching = optimiser_parameters.niching + [score]

                optimized_graph = res_opt
                optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
                optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
                structure = opt_graph_to_bamt(optimized_network)
                optimized_graph.show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(count) + '.png')) 
                OF=round(metric(optimized_network, data=discretized_data)[0],2)
                Score_true = round(metric_str(true_net, data=discretized_data)[0],3)
                SHD=precision_recall(structure, true_net)['SHD']
                SHD_BAMT=precision_recall(structure_BAMT, true_net)['SHD']    
                
                pdf.add_page()
                pdf.set_font("Arial", size = 14)
                pdf.cell(150, 5, txt = str(OF), ln = 1, align = 'C')
                pdf.cell(150, 5, txt = "pop_size = " + str(requirements.pop_size), ln = 1)
                pdf.cell(150, 5, txt = "mutation_prob = " + str(requirements.mutation_prob), ln = 1)
                pdf.cell(150, 5, txt = "crossover_prob = " + str(requirements.crossover_prob), ln = 1)            
                pdf.cell(150, 5, txt = "genetic_scheme_type = " + str(optimiser_parameters.genetic_scheme_type), ln = 1)
                pdf.cell(150, 5, txt = "selection_types = " + str(optimiser_parameters.selection_types), ln = 1)
                pdf.multi_cell(180, 5, txt = "mutation_types = " + str(optimiser_parameters.mutation_types))
                pdf.multi_cell(180, 5, txt = "crossover_types = " + str(optimiser_parameters.crossover_types))
                pdf.cell(150, 5, txt = "stopping_after_n_generation = " + str(optimiser_parameters.stopping_after_n_generation), ln = 1)
                pdf.cell(150, 5, txt = "actual_generation_num = " + str(optimiser.generation_num), ln = 1)     
                pdf.cell(150, 5, txt = "timout = " + str(j), ln = 1)          
                pdf.image('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(count) + '.png',w=165, h=165)
                pdf.multi_cell(180, 5, txt = str(structure))
                pdf.multi_cell(180, 5, txt = 'SHD = '+str(SHD))
                pdf.multi_cell(180, 5, txt = 'SHD_BAMT = '+str(SHD_BAMT))
                pdf.multi_cell(180, 5, txt = 'Score true = ' + str(Score_true))
                pdf.multi_cell(180, 5, txt = 'Score BAMT = ' + str(Score_BAMT))
                pdf.multi_cell(180, 5, txt = 'time = ' + str(elapsed_time))

            
            print('_______________________________________________________________')

            initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]
            init=deepcopy(initial[0])
            
            initial=[]            
            initial.append(res_opt)

            
            for i in range(0, pop_size-1):
                rand = randint(1, 2*len(vertices))
                fl1 = False
                while not fl1:
                    try:
                        fl1 = False
                        g=deepcopy(init)
                        for _ in range(rand):
                            g=deepcopy(custom_mutation_add(g))
                        mylist = []
                        for rule_func in rules:
                            mylist.append(rule_func(g))
                        fl1=all(mylist)
                        if fl1 and len(initial) != 0:
                            # check=[]
                            for i in initial:
                                if check_iequv(g, i):
                                    fl1 = False
                                    break
                    except:
                        pass
                initial.append(g)
            
            
            optimiser = EvoGraphOptimiser(
                graph_generation_params=graph_generation_params,
                metrics=[],
                parameters=optimiser_parameters,
                requirements=requirements, initial_graph=initial,
                log=default_log(logger_name='Bayesian', verbose_level=1))      



            elapsed_time =(time.perf_counter() - start_time)/60
            if last==score:
                l_n+=1
            else:
                last=score
                l_n = 0
            
            
        optimized_graph = res_opt
        optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
        optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
        structure = opt_graph_to_bamt(optimized_network)
        optimized_graph.show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(count) + '.png')) 
        OF=round(metric(optimized_network, data=discretized_data)[0],2)
        exp.append(OF)
        print(OF)
        print('Score BAMT = ', Score_BAMT)
        Score_true = round(metric_str(true_net, data=discretized_data)[0],3)
        print('Score true = ', Score_true)
        print('niching', optimiser_parameters.niching)     
            
    else:

        requirements = PipelineComposerRequirements(
            primary=vertices,
            secondary=vertices, max_arity=100,
            max_depth=100, pop_size=pop_size, num_of_generations=n_generation,
            crossover_prob=crossover_probability, mutation_prob=mutation_probability,
            timeout=timedelta(minutes=time_m)
            )
    
        optimiser_parameters = GPGraphOptimiserParameters(
            # genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            selection_types=[SelectionTypesEnum.tournament],
            # selection_types=[SelectionTypesEnum.tournament_selection_for_MGA],
            mutation_types=mutation_fun,
            crossover_types=crossover_fun,
            # crossover_types=[custom_crossover_parents1, custom_crossover_parents],
            regularization_type=RegularizationTypesEnum.none,
            # если улучшение не происходит в течении ... поколений -> выход
            stopping_after_n_generation=n_generation
        )
        optimiser_parameters.custom = discretized_data
        optimiser_parameters.niching = False

        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
            rules_for_constraint=rules)

        optimiser_parameters.custom = discretized_data
        optimiser = EvoGraphOptimiser(
            graph_generation_params=graph_generation_params,
            metrics=[],
            parameters=optimiser_parameters,
            requirements=requirements, initial_graph=initial,
            log=default_log(logger_name='Bayesian', verbose_level=1))



        start_time = time.perf_counter()
        optimized_graph = optimiser.optimise(partial(metric, data=discretized_data))
        elapsed_time =(time.perf_counter() - start_time)/60         
        optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
        optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
        structure = opt_graph_to_bamt(optimized_network)
        # optimized_graph.show()
        optimized_graph.show(path=('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(count) + '.png')) 
        OF=round(metric(optimized_network, data=discretized_data)[0],3)
        exp.append(OF)
        print(OF)
        print('Score BAMT = ', Score_BAMT)
        Score_true = round(metric_str(true_net, data=discretized_data)[0],3)
        print('Score true = ', Score_true)
        print('time ', elapsed_time)
        
        
    

# ДЛЯ ТАБЛИЦЫ ############################
    def table(network, structure):
    
        final_bn = []
        if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
            final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
        elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
            final_bn = Nets.DiscreteBN()
        elif 'cont' in p.info['types'].values():
            final_bn = Nets.ContinuousBN(use_mixture=False)  

        final_bn.add_nodes(p.info)
        structure = opt_graph_to_bamt(network)
        final_bn.set_structure(edges=structure)
        final_bn.get_info()
        final_bn.fit_parameters(data)

        prediction = dict()

        for c in vertices:
            test = data.drop(columns=[c])
            pred = final_bn.predict(test, 5)
            prediction.update(pred)
        
        result = dict()
        for key, value in prediction.items():
            if node_type[key]=="disc":
                res=round(accuracy_score(data[key], value),2)
            elif node_type[key]=="cont":
                res=round(mean_squared_error(data[key], value, squared=False),2)
            result[key]=res
        
        return result
#######################################

    SHD=precision_recall(structure, true_net)['SHD']
    print('GA_SHD = ', SHD)
 
    SHD_BAMT=precision_recall(structure_BAMT, true_net)['SHD']
    print('BAMT_SHD = ', SHD_BAMT)


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
    pdf.image('C:/Users/Worker1/Documents/FEDOT/examples/V' + str(count) + '.png',w=165, h=165)
    pdf.multi_cell(180, 5, txt = str(structure))
    pdf.multi_cell(180, 5, txt = 'SHD = '+str(SHD))
    pdf.multi_cell(180, 5, txt = 'SHD_BAMT = '+str(SHD_BAMT))
    pdf.multi_cell(180, 5, txt = 'Score true = ' + str(Score_true))
    pdf.multi_cell(180, 5, txt = 'Score BAMT = ' + str(Score_BAMT))
    pdf.multi_cell(180, 5, txt = 'time = ' + str(elapsed_time))
    pdf.multi_cell(180, 5, txt = 'niching = ' + str(optimiser_parameters.niching))
    if nich:
        pdf.multi_cell(180, 5, txt = 'min nich = ' + str(min(optimiser_parameters.niching)))
 
 
 # ТАБЛИЦА ###############
    # pdf.add_page()    
    # pdf.multi_cell(180, 5, txt = 'GA')
    # pdf.set_font("Arial", size = 14)
    # res_GA = table(optimized_network, structure)
    # for heading, row in res_GA.items():
    #     pdf.cell(40, 6, heading, 1)
    #     pdf.cell(40, 6, str(row), 1)
    #     pdf.ln()

    # pdf.add_page()    
    # pdf.multi_cell(180, 5, txt = 'BAMT')
    # pdf.set_font("Arial", size = 14)
    # res_BAMT = table(BAMT_network, structure_BAMT)
    # for heading, row in res_BAMT.items():
    #     pdf.cell(40, 6, heading, 1)
    #     pdf.cell(40, 6, str(row), 1)
    #     pdf.ln()
##############################    

if __name__ == '__main__':
    # data = pd.read_csv(r'examples/data/asia.csv')
    # data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # nodes = list(data.columns)
    #files = ['asia', 'sachs', 'magic-niab', 'ecoli70', 'child']
    # ['earthquake','healthcare','sangiovese']
    # [asia_bnln, sprinkler_bnln, sachs_bnln, alarm_bnln, andes_bnln]
    files = ['sprinkler_bnln']
    nich = False
    micro = False
    pop_size=40
    n_generation = 100
    crossover_probability=0.8
    mutation_probability=0.9
    crossover_fun = [custom_crossover_parents4]
    mutation_fun = [custom_mutation_add, custom_mutation_delete, custom_mutation_reverse]
    max_numb_nich = 10


# files = ['data_asia']
    for file in files:
        data = pd.read_csv('examples/data/'+file+'.csv')
        # data = bn.import_example(data='sachs')
        if file!='credit_card_anomaly' and file!='custom_encoded' and file!='10nodes_cont' and file!='data_asia':
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
            
        # print(data.isna().sum())
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)

        if file == 'mehra-complete':
            for i in data.columns:
                if data[i].dtype.name == 'int64':
                    data = data.astype({i:'float64'})    
        
        with open('examples/data/'+file+'.txt') as f:
            lines = f.readlines()
        true_net = []
        for l in lines:
            e0 = l.split()[0]
            e1 = l.split()[1].split('\n')[0]
            true_net.append((e0, e1))
#         true_net = [("Asia","Tuberculosis"), ("Tuberculosis","Either"), ("Smoking","LungCancer"),
# ("Smoking","Bronchitis"), ("LungCancer","Either"), ("Bronchitis","Dyspnoea"), 
# ("Either","ChestXRay"), ("Either","Dyspnoea")]
        # pdf = FPDF()  
        
        # for j in [1, 2]+list(range(5, 40, 5)):
        
            # pdf = FPDF()
            # time_m=j
            # structure = run_example()
            # pdf.output("f_Experiment_empty_"+file + str(j) + ".pdf")
        try:
            for j in [10]:
                # for exper in range(5,100,5):
                for exper in [1]:                    
                    exp=[]    
                    
                    for count in range(5):
                        time_m=j

                        # metric = custom_metric_cmi_new
                        # metric_str = metric_for_structure_cmi_new

                        # metric = custom_metric_cmi
                        # metric_str = metric_for_structure_cmi

                        metric = custom_metric_LL
                        metric_str = metric_for_structure_LL
                        
                        pdf = FPDF()  
                        structure = run_example()
                    
                        # textfile = open("main_exp_file"+str(exper)+".txt", "w")
                        # textfile.write(str(exp))
                        # textfile.close()  
                    
                        if micro:
                            pdf.output("LL_"+file+"_micro_"+str(count)+str(time.time())+".pdf")
                        else:
                            pdf.output("LL_"+file+"_"+str(count)+str(time.time())+".pdf")
        except Exception as ex:
            print(ex)





    # for i, met in zip(range(1,4), ['K2','BDeu','Bic']):
    #     pdf = FPDF()
    #     for j in range(0):
    #         structure = run_example()
    #     pdf.output("New_custom_metric_asia.pdf")