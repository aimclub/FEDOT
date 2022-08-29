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
parentdir = 'C:\\Users\\anaxa\\Documents\\Projects\\FEDOT'

sys.path.insert(0, parentdir)

import time
from itertools import permutations

from fedot.core.dag.graph import Graph
from joblib import PrintTime

from copy import deepcopy
import itertools
from fedot.core.dag.graph_node import GraphNode
 
import numpy as np
import pandas as pd
import random
from functools import partial
from sklearn import preprocessing
import seaborn as sns
from pyitlib import discrete_random_variable as drv
from statistics import mean

 
import bamt.Preprocessors as pp
from bamt.Builders import StructureBuilder
import bamt.Networks as Nets

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log, Log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
# from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.models import BayesianNetwork as BayesianNetwork_pgmpy
from pgmpy.metrics import structure_score, log_likelihood_score
from pgmpy.estimators import K2Score, BicScore, BDeuScore
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from fpdf import FPDF
from math import log10, ceil
from datetime import timedelta
from random import randint, sample
from pomegranate import *
import networkx as nx
from pgmpy.estimators import MaximumLikelihoodEstimator

class CustomGraphModel(Graph):

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1



class CustomGraphNode(OptNode):
    def __str__(self):
        return self.content["name"]
 

def new_meric_structure_score(graph: CustomGraphModel, data: pd.DataFrame):
    score = 0
    nodes = data.columns.to_list()
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    data_values=data.values
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        struct.append([l1, l2])
   
    
    bn_model = BayesianNetwork_pgmpy(struct)
    bn_model.add_nodes_from(nodes)

    score = structure_score(bn_model, data, scoring_method="k2")

    return [-score]


def new_meric_LL(graph: CustomGraphModel, data: pd.DataFrame):
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
    
    bn_model = BayesianNetwork_pgmpy(struct)
    bn_model.add_nodes_from(data.columns)
    bn_model.fit(data, estimator=MaximumLikelihoodEstimator)
    LL = log_likelihood_score(bn_model, data)

    Dim = 0
    for i in nodes:
        unique = (unique_values[i])
        for j in new_struct[dir_of_vertices[i]]:
            unique = unique * unique_values[dir_of_vertices_rev[j]]
        Dim += unique
    score = LL - (percent*Dim)*log10(len(data))*Dim    

    return [-score]



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
        unique = (unique_values[i])
        for j in new_struct[dir_of_vertices[i]]:
            # надо бы исправить как в формуле (unique - 1)* ...
            unique = unique * unique_values[dir_of_vertices_rev[j]]
        Dim += unique
    score = LL - (percent*Dim)*log10(len(data))*Dim
    return [-score]

# def metric_for_structure_LL(struc, data: pd.DataFrame):
#     score = 0
#     nodes = data.columns.to_list()
#     data_values=data.values
#     struct = struc
    
#     new_struct=[ [] for _ in range(len(vertices))]
#     for pair in struct:
#         i=dir_of_vertices[pair[1]]
#         j=dir_of_vertices[pair[0]]
#         new_struct[i].append(j)
    
#     new_struct=tuple(map(lambda x: tuple(x), new_struct))
#     model = BayesianNetwork.from_structure(data_values, new_struct)
#     model.fit(data_values)
#     model.bake()
#     L=model.log_probability(data_values)
#     LL=L.sum()
#     Dim = 0
#     for i in nodes:
#         unique = unique_values[i]
#         for j in new_struct[dir_of_vertices[i]]:
#             unique = unique * unique_values[dir_of_vertices_rev[j]]
#         Dim += unique
#     score = LL - log10(len(data)/2)*Dim
#     return [-score]    


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
    structure = opt_graph_to_structure(graph)
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

# def metric_for_structure_pass(struc, data: pd.DataFrame):

#     encoder = preprocessing.LabelEncoder()
#     discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
#     p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
#     discretized_data, est = p.apply(data)
    
   


#     final_bn = []
#     if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
#         final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
#     elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
#         final_bn = Nets.DiscreteBN()
#     elif 'cont' in p.info['types'].values():
#         final_bn = Nets.ContinuousBN(use_mixture=False)
    
#     final_bn.add_nodes(p.info)
#     structure = struc
#     # print(structure)
  

#     final_bn.set_structure(edges=structure)
#     final_bn.fit_parameters(data)

#     prediction = dict()
#     for c in vertices:
#         test = data.drop(columns=[c])
#         pred = final_bn.predict(test,5)
#         prediction.update(pred)    


#     result = dict()
#     for key, value in prediction.items():
#         if node_type[key]=="disc":
#             res=1 - round(accuracy_score(data[key], value),2)
#         elif node_type[key]=="cont":
#             res=(round(mean_squared_error(data[key], value, squared=False),2)) / (data[key].max() - data[key].min())
#         result[key]=res
#     score = sum(result.values())
#     return [score]



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

# def metric_for_structure_mi(struc, data: pd.DataFrame):
#     score = 0
#     nodes = data.columns.to_list()
#     data_values=data.values
#     struct = struc

#     new_struct=[ [] for _ in range(len(vertices))]
#     for pair in struct:
#         i=dir_of_vertices[pair[1]]
#         j=dir_of_vertices[pair[0]]
#         new_struct[i].append(j)
    
#     new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
#     def mi(struct, vertices):
#         MI = 0
#         for i in vertices:
#             if i in list(itertools.chain(*struct)):
#                 arcs = [j for j in struct if j[0]==i or j[1]==i]
#                 for a in arcs:
#                     MI += drv.information_mutual(discretized_data[a[0]].values, discretized_data[a[1]].values)
#             else: 
#                 MI += drv.information_mutual(discretized_data[i].values, discretized_data[i].values)
#         return(MI)

#     score = mi(struct, nodes)
    
#     return [score]    


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

# def metric_for_structure_cmi(struc, data: pd.DataFrame):
#     score = 0
#     nodes = data.columns.to_list()
#     data_values=data.values
#     struct = struc

#     new_struct=[ [] for _ in range(len(vertices))]
#     for pair in struct:
#         i=dir_of_vertices[pair[1]]
#         j=dir_of_vertices[pair[0]]
#         new_struct[i].append(j)
    
#     new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
#     def cmi(stru, vertices):
#         CMI = 0
#         for X in vertices:
#             if X in [k[1] for k in stru]:
#                 parents = [j[0] for j in stru if j[1]==X]
#                 if len(parents)==1:
#                     Y=parents[0]
#                     CMI += drv.information_mutual(discretized_data[X].values, discretized_data[Y].values)
#                 else:
#                     for Y in parents:
#                         par = deepcopy(parents)
#                         par.remove(Y)
#                         df_Z_int=deepcopy(discretized_data[par])
#                         df_Z_str = df_Z_int.astype(str)
#                         df_Z = df_Z_str.agg('-'.join, axis=1)
#                         enc = preprocessing.LabelEncoder()
#                         Z = enc.fit_transform(df_Z)
#                         df = deepcopy(discretized_data[[X,Y]])
#                         df.insert(2, 'Z', Z)
#                         CMI += drv.information_mutual_conditional(df[X].values, df[Y].values, df['Z'].values)
#             else: 
#                 CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
#         return(CMI)  


#     score = cmi(struct, nodes)
    
#     return [score]  

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

# def metric_for_structure_cmi_new(struc, data: pd.DataFrame):
#     score = 0
#     nodes = data.columns.to_list()
#     data_values=data.values
#     struct = struc

#     new_struct=[ [] for _ in range(len(vertices))]
#     for pair in struct:
#         i=dir_of_vertices[pair[1]]
#         j=dir_of_vertices[pair[0]]
#         new_struct[i].append(j)
    
#     new_struct=tuple(map(lambda x: tuple(x), new_struct))    
    
#     def cmi(stru, vertices):
#         CMI = 0
#         for X in vertices:
#             if X in [k[1] for k in stru]:
#                 parents = [j[0] for j in stru if j[1]==X]
#                 for Y in parents:
#                     CMI += drv.information_mutual_conditional(discretized_data[X].values, discretized_data[X].values, discretized_data[Y].values)
#             else: 
#                 CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
#         return(CMI)   


#     score = cmi(struct, nodes)
    
#     return [score]  




def custom_metric_DJS(graph: CustomGraphModel, data: pd.DataFrame):

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    N = len(data)
   

    final_bn = []
    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        final_bn = Nets.DiscreteBN()
    elif 'cont' in p.info['types'].values():
        final_bn = Nets.ContinuousBN(use_mixture=False)
    
    final_bn.add_nodes(p.info)
    structure = opt_graph_to_structure(graph)
    final_bn.set_structure(edges=structure)
    final_bn.get_info()
    final_bn.fit_parameters(data)
    sample = final_bn.sample(N) 
    
    score_list = []
    for i in sample.columns:
        sample[i] = sample[i].astype(str).astype(int)
        score_list.append(float(drv.divergence_jensenshannon(data[i].values,sample[i].values)))

    mean_score = mean(score_list)
    return [mean_score]



def opt_graph_to_structure(graph: CustomGraphModel):
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
 

def structure_to_opt_graph(structure):

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    bn = []
    if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
        bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
        bn = Nets.DiscreteBN()
    elif 'cont' in p.info['types'].values():
        bn = Nets.ContinuousBN(use_mixture=False)  

    bn.add_nodes(p.info)
    bn.set_structure(edges=structure)

    fdt = CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                    content={'name': v}) for v in vertices])

    for node in fdt.nodes: 
        parents = []
        for n in bn.nodes:
            if str(node) == str(n):
                parents = n.cont_parents + n.disc_parents
                break
        for n2 in fdt.nodes:
            if str(n2) in parents:
                node.nodes_from.append(n2)
    return fdt


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
 



# Кроссовер происходит между двумя родителями, первый ребенок - копия первого родителя. 
# После этого из второго родителя выбирается узел хотя бы с одним ребром, и все действительные связи, заканчивающиеся на этом узле, 
# добавляются к ребенку. Второй ребенок - копия второго родителя.
def custom_crossover_exchange_parents_one(graph_first, graph_second, max_depth):
    
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]
    
    num_cros = 100
    try:
        for _ in range(num_cros):

            old_edges1 = []
            new_graph_first=deepcopy(graph_first)

            edges = graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node=random.choice(nodes_with_parent_or_child)
                parents=selected_node.nodes_from

                node_from_first_graph=find_node(new_graph_first, selected_node) 
                
                node_from_first_graph.nodes_from=[]
                old_edges1 = new_graph_first.operator.get_all_edges()

                if parents!=[] and parents!=None:
                    parents_in_first_graph=[find_node(new_graph_first, i) for i in parents]
                    for parent in parents_in_first_graph:
                        if [parent, node_from_first_graph] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, node_from_first_graph)

            if old_edges1 != new_graph_first.operator.get_all_edges():
                break    
        
        if old_edges1 == new_graph_first.operator.get_all_edges() and parents!=[] and parents!=None:
            new_graph_first = deepcopy(graph_first)                

    except Exception as ex:
        print(ex)

    return new_graph_first, graph_second


def custom_crossover_exchange_parents_both(graph_first, graph_second, max_depth):
    
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]
    
    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=random.choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=find_node(new_graph_first, selected_node2)
                parents1=selected_node1.nodes_from
                
                selected_node1.nodes_from=[]
                selected_node2.nodes_from=[]
                old_edges1 = new_graph_first.operator.get_all_edges()
                old_edges2 = new_graph_second.operator.get_all_edges()

                if parents2!=[] and parents2!=None:
                    parents_in_first_graph=[find_node(new_graph_first, i) for i in parents2]
                    for parent in parents_in_first_graph:
                        if [parent, selected_node1] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, selected_node1)

                if parents1!=[] and parents1!=None:
                    parents_in_second_graph=[find_node(new_graph_second, i) for i in parents1]
                    for parent in parents_in_second_graph:
                        if [parent, selected_node2] not in old_edges2:
                            new_graph_second.operator.connect_nodes(parent, selected_node2)            


            if old_edges1 != new_graph_first.operator.get_all_edges() or old_edges2 != new_graph_second.operator.get_all_edges():
                break    
        
        # если не получилось добавить новых родителей, тогда оставить изначальный вариант графа
        if old_edges1 == new_graph_first.operator.get_all_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_all_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)       

    except Exception as ex:
        print(ex)
    
    return new_graph_first, new_graph_second


# исправить функцию
def custom_crossover_exchange_parents_deep(graph_first, graph_second, max_depth):
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:

                selected_node2=random.choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=find_node(new_graph_first,selected_node2)
                parents1=selected_node1.nodes_from

                
            selected_node1.nodes_from=[]
            selected_node2.nodes_from=[]
            old_edges1 = new_graph_first.operator.get_all_edges()
            old_edges2 = new_graph_second.operator.get_all_edges()

        
    
            if parents2!=[] and parents2!=None:
                parents_in_first_graph=[find_node(new_graph_first,i) for i in parents2]
                zip_par = list(zip(parents_in_first_graph, parents2))
                for p_i in zip_par:
                    if [p_i[0], selected_node1] not in old_edges1:
                        new_graph_first.operator.connect_nodes(p_i[0], selected_node1)
                        next_par = p_i[1].nodes_from
                        if next_par==[] or next_par==None:
                            continue
                        else:                     
                            next_par_first = [find_node(new_graph_first,i) for i in next_par]
                            for j in next_par_first:
                                last_var_of_graph = deepcopy(new_graph_first)
                                p_i[0].nodes_from=[]                                 
                                if [j, p_i[0]] not in old_edges1:
                                    new_graph_first.operator.connect_nodes(j, p_i[0])   
                                    if [j, p_i[0]] not in new_graph_first.operator.get_all_edges():
                                        new_graph_first = deepcopy(last_var_of_graph)                                    

            if parents1!=[] and parents1!=None:
                parents_in_second_graph=[find_node(new_graph_second,i) for i in parents1]
                zip_par = list(zip(parents_in_second_graph, parents1))                
                for p_i in zip_par:
                    if [p_i[0], selected_node2] not in old_edges2:
                        new_graph_second.operator.connect_nodes(p_i[0], selected_node2)
                        next_par = p_i[1].nodes_from
                        if next_par==[] or next_par==None:
                            continue
                        else:                     
                            next_par_second = [find_node(new_graph_second,i) for i in next_par]
                            for j in next_par_second:
                                last_var_of_graph = deepcopy(new_graph_second)
                                p_i[0].nodes_from=[] 
                                if [j, p_i[0]] not in old_edges2:
                                    new_graph_second.operator.connect_nodes(j, p_i[0])   
                                    if [j, p_i[0]] not in new_graph_second.operator.get_all_edges():
                                        new_graph_second = deepcopy(last_var_of_graph)




        if old_edges1 == new_graph_first.operator.get_all_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_all_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)             

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second


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

        if old_edges1 == new_graph_first.operator.get_all_edges() and new_edges_1!=[] and new_edges_1!=None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_all_edges() and new_edges_2!=[] and new_edges_2!=None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second
# 
#
# # обмен родителями одного узла 
# def custom_crossover_exchange_parents(graph_first: OptGraph, graph_second: OptGraph, max_depth):
#     num_cros = 1

#     def find_node(graph: OptGraph, node):
#         return graph.nodes[dir_of_nodes[node.content['name']]]
#     try:
#         for _ in range(num_cros):
#             new_graph_first = deepcopy(graph_first)
#             new_graph_second = deepcopy(graph_second)      
            
#             rid = random.choice(range(len(new_graph_first.nodes)))
#             node_f = new_graph_first.nodes[rid] 
#             node_s = find_node(new_graph_second, node_f)
#             par_f = node_f.nodes_from
#             par_s = node_s.nodes_from

#             node_f.nodes_from = []
#             node_s.nodes_from = []

#             if par_s!=[] and par_s!=None:
#                 par_f_new = [find_node(new_graph_first, i) for i in par_s]  
#                 for i in par_f_new:
#                     node_f.nodes_from.append(i)

#             if par_f!=[] and par_f!=None:
#                 par_s_new = [find_node(new_graph_second, i) for i in par_f] 
#                 for i in par_s_new:
#                     node_s.nodes_from.append(i)            

#     except Exception as ex:
#         print(ex)
#     return new_graph_first, new_graph_second

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


# I - equivalence
def  check_iequv(ind1, ind2):

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
        # rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    elif 'disc' in types or 'disc_num' in types:
        bn = Nets.DiscreteBN()
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    elif 'cont' in types:
        bn = Nets.ContinuousBN(use_mixture=False)
        rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

    bn.add_nodes(p.info)
        
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score))

    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]

    init=deepcopy(initial[0])

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
    structure_BAMT = opt_graph_to_structure(BAMT_network)
    Score_BAMT = round(metric(BAMT_network, data=discretized_data)[0],6)
    true_fdt = structure_to_opt_graph(true_net)
    Score_true = round(metric(true_fdt, data=discretized_data)[0],6)

    print('bamt score', Score_BAMT)
    print('true score', Score_true)


    from fedot.core.optimisers.adapters import PipelineAdapter
    from fedot.core.optimisers.opt_history import OptHistory
    from fedot.core.utils import fedot_project_root


    # def run_pipeline_and_history_visualization(with_pipeline_visualization=True):
    #     """ Function run visualization of composing history and pipeline """
    #     # Generate pipeline and history
    #     history = OptHistory.load(os.path.join(fedot_project_root(), 'examples', 'data', 'history', 'opt_history.json'))
    #     pipeline = PipelineAdapter().restore(history.individuals[-1][-1].graph)

    #     history.show('fitness_box', pct_best=0.5)
    #     history.show('operations_kde')
    #     history.show('operations_animated_bar', save_path='example_animation.gif', show_fitness=False)
    #     if with_pipeline_visualization:
    #         pipeline.show()

    # run_pipeline_and_history_visualization(history)



    global dir_of_nodes
    dir_of_nodes={initial[0].nodes[i].content['name']:i for i in range(len(initial[0].nodes))} 

    # задать рандомную начальную популяцию. Последнего сделает по _create_randomized_pop_from_inital_graph. 
    # Без этого не выйдет из цикла
    

    # pop_size=100
    # list_of_time=[]
    # start_time = time.perf_counter()

# Если мало узлов, не смотри на эквивлентность
    print('Создание начальной популяции')
    def create_population(pop_size, initial = []):
   
        # генерация рандомных индивидов, соответствующих правилам
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
            
        return initial

    initial = []
    initial = create_population(pop_size, initial) 

    print('Конец создания начальной популяции')

 
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
        'AP': round(corr_undir/pred_len, decimal), 
        'AR': round(corr_undir/true_len, decimal), 
        'AHP': round(corr_dir/pred_len, decimal), 
        'AHR': round(corr_dir/true_len, decimal), 
        'SHD': shd}

    
    def for_pdf(t1, t2):
        return pdf.multi_cell(150, 5, txt = t1 + " = " + str(t2))

    print('Вход')

    global fitness
    global time_passed

    objective = Objective(metric) 
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)
    if sequential:
        start_time = time.perf_counter()
        requirements = PipelineComposerRequirements(
            primary=vertices,
            secondary=vertices, 
            max_arity=100,
            max_depth=100, 
            pop_size=pop_size, 
            num_of_generations=n_generation,
            crossover_prob=crossover_probability, 
            mutation_prob=mutation_probability,
            timeout=timedelta(minutes=time_m)
            )
    
        optimiser_parameters = GPGraphOptimiserParameters(
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            selection_types=[SelectionTypesEnum.tournament],
            mutation_types=mutation_fun,
            crossover_types=crossover_fun,
            regularization_type=RegularizationTypesEnum.none,
            stopping_after_n_generation=stopping_after
        )

        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
            rules_for_constraint=rules)

        optimiser_parameters.custom = discretized_data
        if nich:
            optimiser_parameters.niching = []
        else:
            optimiser_parameters.niching = False
        optimiser = EvoGraphOptimiser(
            graph_generation_params=graph_generation_params,
            parameters=optimiser_parameters,
            requirements=requirements,
            initial_graph=initial,
            objective=objective)


# минуты
        elapsed_time =(time.perf_counter() - start_time)/60
        l_n = 0
        last = 0
        it = 0
        nich_result = []
        nich_list = []
        while l_n <=sequential_count and elapsed_time < time_m and it < max_numb_nich:
            it+=1
            res_opt = optimiser.optimise(objective_eval)[0]
            score = round(metric(res_opt, data=discretized_data)[0],6)
            
            # для seq сохр посл граф и структ
            # optimized_graph = res_opt
            # optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
            # optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
            # structure = opt_graph_to_structure(optimized_network)

            if nich:
                nich_list = optimiser_parameters.niching + [score]
                print(nich_list)
                optimized_graph = res_opt
                optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))                
                optimized_graph.show(path=('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(it-1) + str(crossover_fun[0].__name__)+'.png')) 
                optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
                structure = opt_graph_to_structure(optimized_network)
                OF=round(metric(optimized_network, data=discretized_data)[0],6)
                # SHD=precision_recall(structure, true_net)['SHD']
                
                nich_result = nich_result + [[optimized_graph, structure, OF]]                
                
                # Score_true = round(metric(structure_to_opt_graph(true_net), data=discretized_data)[0],6)
                # SHD_BAMT=precision_recall(structure_BAMT, true_net)['SHD']    
                
                # pdf.add_page()
                # pdf.set_font("Arial", size = 14)
                # pdf.cell(150, 5, txt = str(OF), ln = 1, align = 'C')
                # for_pdf('pop_size', requirements.pop_size)
                # for_pdf('mutation_prob', requirements.mutation_prob)
                # for_pdf('crossover_prob', requirements.crossover_prob)
                # for_pdf('genetic_scheme_type', optimiser_parameters.genetic_scheme_type.name)
                # for_pdf('selection_types', optimiser_parameters.selection_types[0].name)
                # for_pdf('mutation_types', [i.__name__ for i in optimiser_parameters.mutation_types])
                # for_pdf('crossover_types', [i.__name__ for i in optimiser_parameters.crossover_types])
                # for_pdf('stopping_after_n_generation', optimiser_parameters.stopping_after_n_generation)
                # for_pdf('actual_generation_num', optimiser.current_generation_num-1)
                # for_pdf('timeout', time_m)         
                # pdf.image('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(it) + str(crossover_fun[0].__name__)+'.png',w=165, h=165)
                # for_pdf('structure', structure)
                # for_pdf('SHD', SHD)
                # for_pdf('GA',precision_recall(structure, true_net))
                # for_pdf('BAMT',precision_recall(structure_BAMT, true_net))
                # for_pdf('SHD BAMT', SHD_BAMT)
                # for_pdf('Score true', Score_true)


            
            print('_______________________________________________________________')

            initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]
            init=deepcopy(initial[0])
            initial=[] 
            
            if nich:
                initial = deepcopy(create_population(pop_size, initial)) 
            else:      
                initial.append(res_opt)
                initial = deepcopy(create_population(pop_size-1, initial)) 
         

            del requirements
            del optimiser_parameters
            del graph_generation_params
            del optimiser
            requirements = PipelineComposerRequirements(
                primary=vertices,
                secondary=vertices, 
                max_arity=100,
                max_depth=100, 
                pop_size=pop_size, 
                num_of_generations=n_generation,
                crossover_prob=crossover_probability, 
                mutation_prob=mutation_probability,
                timeout=timedelta(minutes=time_m)
                )
        
            optimiser_parameters = GPGraphOptimiserParameters(
                genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
                selection_types=[SelectionTypesEnum.tournament],
                mutation_types=mutation_fun,
                crossover_types=crossover_fun,
                regularization_type=RegularizationTypesEnum.none,
                stopping_after_n_generation=stopping_after
            )

            graph_generation_params = GraphGenerationParams(
                adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
                rules_for_constraint=rules)

            optimiser_parameters.custom = discretized_data

            optimiser_parameters.niching = nich_list

            optimiser = EvoGraphOptimiser(
                graph_generation_params=graph_generation_params,
                parameters=optimiser_parameters,
                requirements=requirements,
                initial_graph=initial,
                objective=objective)

            optimiser = EvoGraphOptimiser(
                graph_generation_params=graph_generation_params,
                parameters=optimiser_parameters,
                requirements=requirements, 
                initial_graph=initial,
                objective=objective)      



            elapsed_time =(time.perf_counter() - start_time)/60
            if last==score:
                l_n+=1
            else:
                last=score
                l_n = 0
            
        if nich:
            # results of sequential niching
            index_min = np.argmin(nich_list)
            optimized_graph = nich_result[index_min][0]
            time_passed = elapsed_time 
            # optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
            # optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
            structure = nich_result[index_min][1]
            # optimized_graph.show(path=('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(index_min) + str(crossover_fun[0].__name__)+ '.png')) 
            OF=nich_result[index_min][2]
            
            
            print('niching', nich_list)
        else:
            # results of sequential 
            optimized_graph = res_opt
            time_passed = elapsed_time 
            optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
            optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
            structure = opt_graph_to_structure(optimized_network)
            optimized_graph.show(path=('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(count) + str(crossover_fun[0].__name__)+ '.png')) 
            OF=round(metric(optimized_network, data=discretized_data)[0],6)


                 
            
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
            stopping_after_n_generation=stopping_after
            # stopping_after_n_generation=10
        )
        optimiser_parameters.custom = discretized_data
        optimiser_parameters.niching = False

        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
            rules_for_constraint=rules)

        optimiser_parameters.custom = discretized_data
        optimiser = EvoGraphOptimiser(
            graph_generation_params=graph_generation_params,
            parameters=optimiser_parameters,
            requirements=requirements,
            initial_graph=initial,
            objective=objective)


        start_time = time.perf_counter()

        optimized_graph = optimiser.optimise(objective_eval)[0]

        elapsed_time =(time.perf_counter() - start_time)/60 
        time_passed = elapsed_time     
        optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
        optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)
        structure = opt_graph_to_structure(optimized_network)

        optimized_graph.show(path=('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(count) + str(crossover_fun[0].__name__)+'.png')) 
        OF=round(metric(optimized_network, data=discretized_data)[0],6)
        
    
    fitness = OF 
    print('Score GA = ', fitness)
    print('Score BAMT = ', Score_BAMT)
    print('Score true = ', Score_true)
        
        
    

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
        structure = opt_graph_to_structure(network)
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

    global shd
    SHD=precision_recall(structure, true_net)['SHD']
    shd = SHD

    SHD_BAMT=precision_recall(structure_BAMT, true_net)['SHD']

    pdf.add_page()
    pdf.set_font("Arial", size = 14)
    pdf.cell(150, 5, txt = str(OF), ln = 1, align = 'C')
    for_pdf('pop_size', requirements.pop_size)
    for_pdf('n_generation', requirements.num_of_generations)
    for_pdf('mutation_prob', requirements.mutation_prob)
    for_pdf('crossover_prob', requirements.crossover_prob)
    for_pdf('genetic_scheme_type', optimiser_parameters.genetic_scheme_type.name)
    for_pdf('selection_types', optimiser_parameters.selection_types[0].name)
    for_pdf('mutation_types', [i.__name__ for i in optimiser_parameters.mutation_types])
    for_pdf('crossover_types', [i.__name__ for i in optimiser_parameters.crossover_types])
    for_pdf('stopping_after_n_generation', optimiser_parameters.stopping_after_n_generation)
    for_pdf('actual_generation_num', optimiser.current_generation_num-1)
    for_pdf('timeout', time_m)
    if nich:
        pdf.image('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(index_min) + str(crossover_fun[0].__name__)+'.png',w=165, h=165)
    else:
        pdf.image('C:/Users/anaxa/Documents/Projects/FEDOT/examples/pictures/V' + str(count) + str(crossover_fun[0].__name__)+'.png',w=165, h=165)
    for_pdf('structure', structure)
    for_pdf('SHD', SHD)
    for_pdf('SHD BAMT', SHD_BAMT)
    for_pdf('Score true', Score_true)
    for_pdf('Score BAMT', Score_BAMT)
    for_pdf('time', time_passed)
    for_pdf('sequential', sequential)
    for_pdf('nich', nich)
    if nich:
        for_pdf('niching', nich_list)
        for_pdf('min nich', min(nich_list))    

    for_pdf('GA',precision_recall(structure, true_net))
    for_pdf('BAMT',precision_recall(structure_BAMT, true_net))
 
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
    #files = ['asia', 'sachs', 'magic-niab', 'ecoli70', 'child']
    # ['earthquake','healthcare','sangiovese','cancer']
    # [asia_bnln, sachs_bnln, sprinkler_bnln, alarm_bnln, andes_bnln]
    
    files = ['earthquake']


    sequential = True
    sequential_count = 5    
    nich = False
    max_numb_nich = 100
    pop_size = 40
    n_generation = 100
    crossover_probability = 0.8
    mutation_probability = 0.9
    stopping_after = 10
    mutation_fun = [custom_mutation_add, custom_mutation_delete, custom_mutation_reverse]
    crossover_funs = [[
        custom_crossover_exchange_edges, 
    custom_crossover_exchange_parents_one, 
    custom_crossover_exchange_parents_both
    ]]


    for file in files:
        percent = 0.02    
        data = pd.read_csv('examples/data/'+file+'.csv')
        if file!='credit_card_anomaly' and file!='custom_encoded' and file!='10nodes_cont' and file!='data_asia':
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)
        print(data.columns)

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

        
        try:
            for selected_crossover in crossover_funs:
                if type(selected_crossover) == list:
                    crossover_fun = selected_crossover 
                else:
                    crossover_fun = [selected_crossover] 
                for count in range(10):
                    time_m=10

                    # [custom_metric_LL, custom_metric_pass, custom_metric_mi, 
                    # [custom_metric_cmi, custom_metric_cmi_new, custom_metric_DJS]
                    # [new_meric_structure_score]

                    metric = custom_metric_LL

                    pdf = FPDF()  
                    
                    structure = run_example()          
                    if type(crossover_funs[0]) == list:
                        textfile = open("C:/Users/anaxa/Desktop/article/" + 
                    file+'_p_'+str(percent)+"_LL_"+"sequential_"+str(sequential)+"_nich_"+str(nich)+'_custom_crossover_all'+".txt", "a")
                    else:
                        textfile = open("C:/Users/anaxa/Desktop/article/" + 
                    file+'_p_'+str(percent)+"_LL_"+"sequential_"+str(sequential)+"_nich_"+str(nich)+"_"+str(crossover_fun[0].__name__)+".txt", "a")
                    
                    textfile.write(str(fitness)+';'+str(shd)+';'+str(time_passed))
                    textfile.write('\n')
                    textfile.close()  
                
                    if sequential:
                        if type(crossover_funs[0]) == list:
                            pdf.output("C:/Users/anaxa/Desktop/article/" +
                        'p_'+str(percent)+"_LL_"+file+'_'+str(count)+"_sequential_"+str(sequential)+"_nich_"+str(nich)+'_custom_crossover_all'+".pdf")
                        else:                        
                            pdf.output("C:/Users/anaxa/Desktop/article/" +
                        'p_'+str(percent)+"_LL_"+file+'_'+str(count)+"_sequential_"+str(sequential)+"_nich_"+str(nich)+"_"+str(crossover_fun[0].__name__)+".pdf")
                    else:
                        if type(crossover_funs[0]) == list:
                            pdf.output("C:/Users/anaxa/Desktop/article/" +
                        'p_'+str(percent)+"_LL_"+file+"_"+str(count)+'_custom_crossover_all'+".pdf")
                        else:
                            pdf.output("C:/Users/anaxa/Desktop/article/" +
                        'p_'+str(percent)+"_LL_"+file+"_"+str(count)+"_"+str(crossover_fun[0].__name__)+".pdf")
        except Exception as ex:
            print(ex)

