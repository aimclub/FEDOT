
from datetime import timedelta
import sys
from typing import Optional, Union, List
import os
import time

parentdir = os.getcwd()




# parentdir = 'C:\\Users\\anaxa\\Documents\\Projects\\\CompositeBayesianNetworks\\FEDOT'
sys.path.insert(0, parentdir)
from fedot.core.dag.graph import Graph
from copy import deepcopy
import pandas as pd
from random import choice, sample
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
from fedot.core.operations.evaluation.evaluation_interfaces import SkLearnEvaluationStrategy
from fedot.core.data.data import InputData, OutputData, process_target_and_features
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score
from math import ceil
from examples.composite_model import CompositeModel
from examples.composite_node import CompositeNode
import bamt.Networks as Nets
from scipy.stats import norm
from numpy import std, mean, log, dot
from sklearn.metrics import mean_squared_error
from itertools import chain, starmap
from math import log10


def composite_metric(graph: CompositeModel, data: pd.DataFrame):
    score, len_data, edges_count = 0, len(data), len(graph.get_edges())
    for node in graph.nodes:
        data_of_node = data[node.content['name']]
        if node.nodes_from == None or node.nodes_from == []:
            if node.content['type'] == 'cont':
                mu, sigma = mean(data_of_node), std(data_of_node)
                score += norm.logpdf(data_of_node.values, loc=mu, scale=sigma).sum()
            else:
                # if node.content['type'] == 'disc' or node.content['type'] == 'disc_num'
                count = data_of_node.value_counts().values
                frequency  = log(count / len_data)
                score += dot(count, frequency)                
        else:
            model, columns, target, idx = node.content['parent_model'], [n.content['name'] for n in node.nodes_from], data_of_node.to_numpy(), data.index.to_numpy()
            features = data[columns].to_numpy()
            if node.content['type'] == 'cont':
                task = Task(TaskTypesEnum.regression)            
            else:
                task = Task(TaskTypesEnum.classification)
            data_type = DataTypesEnum.table
            train = InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)
            fitted_model = model.fit(train)
            
            if node.content['type'] == 'cont':
                predict = fitted_model.predict(train.features)
                mse =  mean_squared_error(target, predict,squared=False)
                score += norm.logpdf(target, loc=predict, scale=mse).sum()
            else:
                # if node.content['type'] == 'disc' or node.content['type'] == 'disc_num'
                predict_proba = fitted_model.predict_proba(features)
                # sum(starmap(f, zip(predict_proba, target)))
                score += sum([log(predict_proba[i][target[i]]) for i in idx])
    score -= len(graph.nodes)*log10(len_data)*edges_count/4

    return [-score]


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

    dir_of_nodes={graph_first.nodes[i].content['name']:i for i in range(len(graph_first.nodes))}     
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
            choice_edges_1 = sample(edges_1, count)
            choice_edges_2 = sample(edges_2, count)
            
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

def custom_crossover_exchange_parents_both(graph_first, graph_second, max_depth):

    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]

    dir_of_nodes={graph_first.nodes[i].content['name']:i for i in range(len(graph_first.nodes))}     
    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_edges()
            flatten_edges = list(chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1=find_node(new_graph_first, selected_node2)
                parents1=selected_node1.nodes_from
                
                selected_node1.nodes_from=[]
                selected_node2.nodes_from=[]
                old_edges1 = new_graph_first.operator.get_edges()
                old_edges2 = new_graph_second.operator.get_edges()

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


            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break    
        
        # если не получилось добавить новых родителей, тогда оставить изначальный вариант графа
        if old_edges1 == new_graph_first.operator.get_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)       

    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_all_model(graph_first: OptGraph, graph_second: OptGraph, max_depth):
    def find_node(graph: OptGraph, node):
        return graph.nodes[dir_of_nodes[node.content['name']]]

    dir_of_nodes={graph_first.nodes[i].content['name']:i for i in range(len(graph_first.nodes))}     
    
    num_cros = 100
    try:
        for _ in range(num_cros):
            selected_node1=choice(graph_first.nodes)
            if selected_node1.nodes_from == None or selected_node1.nodes_from == []:
                continue
            
            selected_node2=find_node(graph_second, selected_node1)
            if selected_node2.nodes_from == None or selected_node2.nodes_from == []:
                continue            

            model1 = selected_node1.content['parent_model']
            model2 = selected_node2.content['parent_model']

            selected_node1.content['parent_model'] = model2
            selected_node2.content['parent_model'] = model1

            break

    except Exception as ex:
        print(ex)
    return graph_first, graph_second


# Структурные мутации
# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add_structure(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
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
 

def custom_mutation_delete_structure(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
                break
    except Exception as ex:
        print(ex) 
    return graph


def custom_mutation_reverse_structure(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)   
                break         
    except Exception as ex:
        print(ex)  
    return graph




# parent_model
# parent_model_regr = ['xgbreg','adareg','gbr','dtreg','treg','rfr','linear',
# 'ridge','lasso','svr','sgdr','lgbmreg', 'catboostreg']
# parent_model_class = ['xgboost','logit','bernb','multinb','dt','rf',
# 'mlp','lgbm', 'catboost']

parent_model_regr = ['xgbreg','linear',
    'ridge','lasso', 'catboostreg'
    ]
parent_model_class = ['xgboost','logit',
    'catboost'
    ]

def random_choice_model(node_type):
    if node_type == 'cont':
        return SkLearnEvaluationStrategy(choice(parent_model_regr))
    else:
        return SkLearnEvaluationStrategy(choice(parent_model_class))



def custom_mutation_add_model(graph: OptGraph, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [node for node in all_nodes if (node.nodes_from!=[] and node.nodes_from!=None)]
        if nodes_with_parents == []:
            return graph
        node = choice(nodes_with_parents)
        node.content['parent_model'] = random_choice_model(node.content['type'])
    except Exception as ex:
        print(ex)  
    return graph

def mutation_set1(graph: OptGraph, **kwargs):
    return custom_mutation_add_model(custom_mutation_add_structure(graph, **kwargs))

def mutation_set2(graph: OptGraph, **kwargs):
    return custom_mutation_add_model(custom_mutation_delete_structure(graph, **kwargs))
        
def mutation_set3(graph: OptGraph, **kwargs):
    return custom_mutation_add_model(custom_mutation_reverse_structure(graph, **kwargs))



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
    # p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    p = pp.Preprocessor([('encoder', encoder)])
    p2 = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, _ = p.apply(data)
    discretized_data2, _ = p2.apply(data)

    # словарь: {имя_узла: уникальный_номер_узла}
    global dir_of_nodes
    dir_of_nodes={data.columns[i]:i for i in range(len(data.columns))}     

    # правила для байесовских сетей: нет петель, нет циклов, нет повторяющихся узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    

    # задаем для оптимизатора fitness-функцию
    objective = Objective(composite_metric) 
    objective_eval = ObjectiveEvaluate(objective, data = discretized_data)    


    # one_set_models = {}
    # for v in vertices:
    #     if p.nodes_types[v] == 'cont':
    #         one_set_models[v] = choice(parent_model_regr)
    #     else:
    #         one_set_models[v] = choice(parent_model_class)

    # one_set_models = {
    #     'A':'bernb',
    #     'C':'mlp',
    #     'D':'ridge',
    #     'H':'catboost', 
    #     'I':'ridge',
    #     'O':'adareg',
    #     'T':'xgbreg'
    # }

    # инициализация начальной сети (пустая)
    # initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
    #                                                 content={'name': vertex}, 
    #                                                 type = dir_of_nodes[vertex]) for vertex in vertices])]


    # initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
    #                                                 content={'name': vertex,
    #                                                 'type': p.nodes_types[vertex],
    #                                                 'parent_model': SkLearnEvaluationStrategy(choice(parent_model_regr if p.nodes_types[vertex] == 'cont' else parent_model_class))}) 
    #                                                 for vertex in vertices])]    
    # initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
    #                                                 content={'name': vertex,
    #                                                 'type': p.nodes_types[vertex],
    #                                                 'parent_model': SkLearnEvaluationStrategy(one_set_models[vertex])}) 
    #                                                 for vertex in vertices])]       
    initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                    content={'name': vertex,
                                                            'type': p.nodes_types[vertex],
                                                            'parent_model': None}) 
                                                    for vertex in vertices])] 
    init = initial[0]
    # добавим для начального графа три ребра
    # init = custom_mutation_add_structure(custom_mutation_add_structure(custom_mutation_add_structure(init)))

    def structure_to_opt_graph(fdt, structure):

        encoder = preprocessing.LabelEncoder()
        # discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('encoder', encoder)])
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


    # заполнить пустую сети CompositeModel
    # for node in init.nodes: 
    #     parents = []
    #     for n in bn.nodes:
    #         if str(node) == str(n):
    #             parents = n.cont_parents + n.disc_parents
    #             break
    #     for n2 in init.nodes:
    #         if str(n2) in parents:
    #             node.nodes_from.append(n2)
    
# назначить parent_model узлам с родителями    
    for node in init.nodes:
        if not (node.nodes_from == None or node.nodes_from == []):
            node.content['parent_model'] = random_choice_model(node.content['type'])   

    
#     # score true and bamt
#     init = structure_to_opt_graph(init, [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')]
#     )
# для эталонной структуры и bamt    
    # for node in init.nodes:
    #     if not (node.nodes_from == None or node.nodes_from == []):
    #         if node.content['type'] == 'cont':
    #             node.content['parent_model'] = SkLearnEvaluationStrategy('linear')
    #         else:
    #             node.content['parent_model'] = SkLearnEvaluationStrategy('logit')       
#     print('score_true', composite_metric(init, discretized_data))
    
    def f(m):
        if m == None:
            return None
        else:
            return (node.content['parent_model'].implementation_info)

    # score = composite_metric(init, discretized_data)
    # [print(node.content['name'],node.content['type'], node.content['parent_model'].operation_impl) for node in init.nodes]
    # print(score)
    # init.show()
    
    # вывод parent_model по узлам
    for node in init.nodes:
        print(node.content['name'], node.content['type'], f(node.content['parent_model']))
    # print('score_true', composite_metric(init, discretized_data))

    def bamt_sore():
        initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                        content={'name': vertex,
                                                                'type': p.nodes_types[vertex],
                                                                'parent_model': None}) 
                                                        for vertex in vertices])] 
        init = initial[0]
        
        types=list(p.info['types'].values())
        if 'cont' in types and ('disc' in types or 'disc_num' in types):
            bn = Nets.HybridBN(has_logit=False, use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'disc' in types or 'disc_num' in types:
            bn = Nets.DiscreteBN()
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
        elif 'cont' in types:
            bn = Nets.ContinuousBN(use_mixture=False)
            rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]

        bn.add_nodes(p.info)
        bn.add_edges(discretized_data2, scoring_function=('K2', K2Score))        

        for node in init.nodes: 
            parents = []
            for n in bn.nodes:
                if str(node) == str(n):
                    parents = n.cont_parents + n.disc_parents
                    break
            for n2 in init.nodes:
                if str(n2) in parents:
                    node.nodes_from.append(n2)

        for node in init.nodes:
            if not (node.nodes_from == None or node.nodes_from == []):
                if node.content['type'] == 'cont':
                    node.content['parent_model'] = SkLearnEvaluationStrategy('linear')
                else:
                    node.content['parent_model'] = SkLearnEvaluationStrategy('logit')    
    
        # def child_dict(net: list):
        #     res_dict = dict()
        #     for e0, e1 in net:
        #         if e1 in res_dict:
        #             res_dict[e1].append(e0)
        #         else:
        #             res_dict[e1] = [e0]
        #     return res_dict

        # def precision_recall(pred, true_net: list, decimal = 2):

        #     edges= pred.get_edges()
        #     struct = []
        #     for s in edges:
        #         struct.append((s[0].content['name'], s[1].content['name']))

        #     pred_net = deepcopy(struct)

        #     pred_dict = child_dict(pred_net)
        #     true_dict = child_dict(true_net)
        #     corr_undir = 0
        #     corr_dir = 0
        #     for e0, e1 in pred_net:
        #         flag = True
        #         if e1 in true_dict:
        #             if e0 in true_dict[e1]:
        #                 corr_undir += 1
        #                 corr_dir += 1
        #                 flag = False
        #         if (e0 in true_dict) and flag:
        #             if e1 in true_dict[e0]:
        #                 corr_undir += 1
        #     pred_len = len(pred_net)
        #     true_len = len(true_net)
        #     shd = pred_len + true_len - corr_undir - corr_dir
        #     return {
        #     'SHD': shd}
        
        # true_net = [('Erk', 'Akt'), ('Mek', 'Erk'), ('PIP3', 'PIP2'), ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'Mek'), ('PKA', 'P38'), ('PKA', 'Raf'), ('PKC', 'Jnk'), ('PKC', 'Mek'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKC', 'Raf'), ('Plcg', 'PIP2'), ('Plcg', 'PIP3'), ('Raf', 'Mek')]  
        # SHD = precision_recall(init, true_net)['SHD']       
        # print(SHD)

       
        return print('score_bamt', composite_metric(init, discretized_data))
    
    
    def true_sore():
        initial = [CompositeModel(nodes=[CompositeNode(nodes_from=None,
                                                        content={'name': vertex,
                                                                'type': p.nodes_types[vertex],
                                                                'parent_model': None}) 
                                                        for vertex in vertices])] 
        init = initial[0]
        
        dict_true_str = {'asia':
        [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')],

        'cancer':
        [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')],

        'earthquake':
        [('Burglary', 'Alarm'), ('Earthquake', 'Alarm'), ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')],

        'sachs':
        [('Erk', 'Akt'), ('Mek', 'Erk'), ('PIP3', 'PIP2'), ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'Mek'), ('PKA', 'P38'), ('PKA', 'Raf'), ('PKC', 'Jnk'), ('PKC', 'Mek'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKC', 'Raf'), ('Plcg', 'PIP2'), ('Plcg', 'PIP3'), ('Raf', 'Mek')],  

        'healthcare':
        [('A', 'C'), ('A', 'D'), ('A', 'H'), ('A', 'O'), ('C', 'I'), ('D', 'I'), ('H', 'D'), ('I', 'T'), ('O', 'T')]}     
        init = structure_to_opt_graph(init, dict_true_str[file])
        
        for node in init.nodes:
            if not (node.nodes_from == None or node.nodes_from == []):
                if node.content['type'] == 'cont':
                    node.content['parent_model'] = SkLearnEvaluationStrategy('linear')
                else:
                    node.content['parent_model'] = SkLearnEvaluationStrategy('logit')    
    
        return print('score_true', composite_metric(init, discretized_data))    
    

    requirements = PipelineComposerRequirements(
        primary=vertices,
        secondary=vertices, 
        max_arity=100,
        max_depth=100, 
        num_of_generations=n_generation,
        timeout=timedelta(minutes=time_m)
        )

    optimiser_parameters = GPGraphOptimizerParameters(
        pop_size=pop_size,
        crossover_prob=crossover_probability, 
        mutation_prob=mutation_probability,
        genetic_scheme_type = GeneticSchemeTypesEnum.steady_state,
        selection_types = [SelectionTypesEnum.tournament],
        mutation_types = [
        custom_mutation_add_structure, 
        custom_mutation_delete_structure, 
        custom_mutation_reverse_structure, 
        custom_mutation_add_model
        ],

        crossover_types = [
            custom_crossover_exchange_edges,
            custom_crossover_all_model,
            custom_crossover_exchange_parents_both
            ]
    )

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CompositeModel, base_node_class=CompositeNode),
        rules_for_constraint=rules)

    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=[init],
        objective=objective)


    def connect_nodes(self, parent: CompositeNode, child: CompositeNode):
        if child.descriptive_id not in [p.descriptive_id for p in parent.ordered_subnodes_hierarchy()]:
            try:
                if child.nodes_from==None or child.nodes_from==[]:
                    child.nodes_from=[]
                    child.content['parent_model'] = random_choice_model(child.content['type'])                     
                child.nodes_from.append(parent)
            except Exception as ex:
                print(ex)
    
    def disconnect_nodes(self, node_parent: CompositeNode, node_child: CompositeNode,
                        clean_up_leftovers: bool = True):
        if not node_child.nodes_from or node_parent not in node_child.nodes_from:
            return
        elif node_parent not in self._nodes or node_child not in self._nodes:
            return
        elif len(node_child.nodes_from) == 1:
            node_child.nodes_from = None
        else:
            node_child.nodes_from.remove(node_parent)

        if clean_up_leftovers:
            self._clean_up_leftovers(node_parent)

        self._postprocess_nodes(self, self._nodes)

        if node_child.nodes_from == [] or node_child.nodes_from == None:
            node_child.content['parent_model'] = None



    def reverse_edge(self, node_parent: CompositeNode, node_child: CompositeNode):
        self.disconnect_nodes(node_parent, node_child, False)
        self.connect_nodes(node_child, node_parent)

    GraphOperator.reverse_edge = reverse_edge
    GraphOperator.connect_nodes = connect_nodes
    GraphOperator.disconnect_nodes = disconnect_nodes
 
    
    # запуск оптимизатора
    optimized_graph = optimiser.optimise(objective_eval)[0]
    # вывод полученного графа
    # for n in optimized_graph.nodes:
    #     n.content['parent_model']

    # вывод parent_model финального графа
    for node in optimized_graph.nodes:
        print(node.content['name'],node.content['type'], f(node.content['parent_model']))

    # optimized_graph.show()


if __name__ == '__main__':

    # файл с исходными данными (должен лежать в 'examples/data/')
    file = 'healthcare'
    # размер популяции 
    pop_size = 20
    # количество поколений
    n_generation = 100
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 100
    run_example() 