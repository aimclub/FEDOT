# import pandas as pd
# data = pd.read_csv(r'examples/data/alarm.csv')
# data.drop(['Unnamed: 0'], axis=1, inplace=True)
# print(data) 
# data = pd.read_csv(r'examples/data/mehra-complete.csv')
# data.drop(['Unnamed: 0'], axis=1, inplace=True)
# print(data)

from lib2to3.pytree import Node, type_repr
from re import I
from sre_parse import State
import sys
from typing import Optional, Union, List
parentdir = 'C:\\Users\\Worker1\\Documents\\FEDOT'
sys.path.insert(0, parentdir)

from fedot.core.dag.graph import Graph
from joblib import PrintTime

import numpy as np
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


# import math
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
    score=L.sum()
    return [-score]

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

data = pd.read_csv(r'examples/data/asia.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

vertices = list(data.columns)
dir_of_vertices={vertices[i]:i for i in range(len(vertices))}
encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data)
print(discretized_data)

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


# OF=round(custom_metric(initial[0], data=discretized_data)[0],2)
# initial[0].show()

# network = BayesianNetwork("Graph_name")
# e=network.graph

d1 = DiscreteDistribution({'A': 0.2, 'B': 0.8})
d2 = DiscreteDistribution({'A': 0.6, 'B': 0.4})
# d2 = ConditionalProbabilityTable([['A', 'A', 0.1],
#                                         ['A', 'B', 0.9],
#                                         ['B', 'A', 0.4],
#                                         ['B', 'B', 0.6]], [d1])
s1 = Node( d1, name="s1" )
s2 = Node( d2, name="s2" )
model = BayesianNetwork()
model.add_nodes(s1, s2)

model.bake()
L=model.log_probability(np.array([['A', 'A'],['B', 'B']]))













# for node in initial[0].nodes: 
#     parents = []
#     for n in bn.nodes:
#         if str(node) == str(n):
#             parents = n.cont_parents + n.disc_parents
#             break
#     for n2 in initial[0].nodes:
#         if str(n2) in parents:
#             node.nodes_from.append(n2)
                





# asia
# ['smoke', 'tub', 'asia', 'dysp', 'xray', 'either', 'bronc', 'lung']

# sachs
# ["Akt","Erk","Jnk","Mek","P38","PIP2","PIP3","PKA","PKC","Plcg","Raf"]

# magic_niab
# ["YR.GLASS","HT","YR.FIELD","MIL","FT","G418","G311","G1217","G800","G866","G795","G2570","G260","G2920","G832","G1896","G2953","G266","G847","G942","G200","G257","G2208","G1373","G599","G261","G383","G1853","G1033","G1945","G1338","G1276","G1263","G1789","G2318","G1294","G1800","YLD","FUS","G1750","G524","G775","G2835","G43"]

# ecoli70
# ["aceB","asnA","atpD","atpG","b1191","b1583","b1963","cchB","cspA","cspG","dnaG","dnaJ","dnaK","eutG","fixC","flgD","folK","ftsJ","gltA","hupB","ibpB","icdA","lacA","lacY","lacZ","lpdA","mopB","nmpC","nuoM","pspA","pspB","sucA","sucD","tnaA","yaeM","yceP","ycgX","yecO","yedE","yfaD","yfiA","ygbD","ygcE","yhdM","yheI","yjbO"]

# child
# ["BirthAsphyxia","HypDistrib","HypoxiaInO2","CO2","ChestXray","Grunting","LVHreport","LowerBodyO2","RUQO2","CO2Report","XrayReport","Disease","GruntingReport","Age","LVH","DuctFlow","CardiacMixing","LungParench","LungFlow","Sick"]






import numpy as np

vertices=['smoke', 'tub', 'asia', 'dysp', 'xray', 'either', 'bronc', 'lung']
matrix=np.zeros((len(vertices),len(vertices)))

d = dict()
for i in range(len(vertices)):
    d[vertices[i]]=i
print(d)
my_structures=[


]
for s in my_structures:
    for e in s:
        i = d[e[0]]
        j=  d[e[1]]
        matrix[i,j] += 1

print(matrix/len(my_structures))
