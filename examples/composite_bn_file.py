
from datetime import timedelta
import sys
from typing import Optional, Union, List
import os


parentdir = os.getcwd()




# parentdir = 'C:\\Users\\anaxa\\Documents\\Projects\\\CompositeBayesianNetworks\\FEDOT'
sys.path.insert(0, parentdir)
from fedot.core.dag.graph import Graph
from copy import deepcopy
import pandas as pd
import numpy as np
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
# from scipy.stats import variation
from numpy import std, mean
from sklearn.metrics import mean_squared_error


def composite_metric(graph: CompositeModel, data: pd.DataFrame):
    score = 0
    len_data = len(data)
    for node in graph.nodes:
        if node.nodes_from == None or node.nodes_from == []:
            if node.content['type'] == 'disc' or node.content['type'] == 'disc_num':
                count = data[node.content['name']].value_counts().values
                frequency  =  count / len_data
                score += np.log(np.dot(count, frequency))
            if node.content['type'] == 'cont':
                mu = mean(data[node.content['name']])
                sigma = std(data[node.content['name']])
                score += norm.logpdf(data[node.content['name']], loc=mu, scale=sigma).sum()
        else:
            model = node.content['parent_model']
            columns = [n.content['name'] for n in node.nodes_from]
            features = data[columns].to_numpy()
            target = data[node.content['name']].to_numpy()
            idx = data.index.to_numpy()
            if node.content['type'] == 'disc':
                task = Task(TaskTypesEnum.classification)
            elif node.content['type'] == 'cont':
                task = Task(TaskTypesEnum.regression)
            data_type = DataTypesEnum.table
            train = InputData(idx=idx, features=features, target=target, task=task, data_type=data_type)
            fitted_model = model.fit(train)
            
            if node.content['type'] == 'cont':
                predict = fitted_model.predict(train.features)
                mu = mean(predict)
                mse = mean_squared_error(target, predict,squared=False)
                score += norm.logpdf(predict, loc=mu, scale=mse).sum()

            elif node.content['type'] == 'disc' or node.content['type'] == 'disc_num':
                predict_proba = fitted_model.predict_proba(features)
                score += sum([predict_proba[i][target[i]] for i in idx])
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


# Структурные мутации
# задаем три варианта мутации: добавление узла, удаление узла, разворот узла
def custom_mutation_add_structure(graph: OptGraph, **kwargs):
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
 

def custom_mutation_delete_structure(graph: OptGraph, **kwargs):
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


def custom_mutation_reverse_structure(graph: OptGraph, **kwargs):
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
        return SkLearnEvaluationStrategy(random.choice(parent_model_regr))
    else:
        return SkLearnEvaluationStrategy(random.choice(parent_model_class))



def custom_mutation_add_model(graph: OptGraph, **kwargs):
    try:
        all_nodes = graph.nodes
        nodes_with_parents = [node for node in all_nodes if (node.nodes_from!=[] and node.nodes_from!=None)]
        node = random.choice(nodes_with_parents)
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
    #         one_set_models[v] = random.choice(parent_model_regr)
    #     else:
    #         one_set_models[v] = random.choice(parent_model_class)

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
    #                                                 'parent_model': SkLearnEvaluationStrategy(random.choice(parent_model_regr if p.nodes_types[vertex] == 'cont' else parent_model_class))}) 
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


    # найдет сеть по K2
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

    # def structure_to_opt_graph(fdt, structure):

    #     encoder = preprocessing.LabelEncoder()
    #     # discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    #     p = pp.Preprocessor([('encoder', encoder)])
    #     discretized_data, est = p.apply(data)

    #     bn = []
    #     if 'cont' in p.info['types'].values() and ('disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values()):
    #         bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    #     elif 'disc' in p.info['types'].values() or 'disc_num' in p.info['types'].values():
    #         bn = Nets.DiscreteBN()
    #     elif 'cont' in p.info['types'].values():
    #         bn = Nets.ContinuousBN(use_mixture=False)  

    #     bn.add_nodes(p.info)
    #     bn.set_structure(edges=structure)
        
    #     for node in fdt.nodes: 
    #         parents = []
    #         for n in bn.nodes:
    #             if str(node) == str(n):
    #                 parents = n.cont_parents + n.disc_parents
    #                 break
    #         for n2 in fdt.nodes:
    #             if str(n2) in parents:
    #                 node.nodes_from.append(n2)        
        
    #     return fdt    

    # init = structure_to_opt_graph(init, [('D','A'),('D','T'),('H','C'),('A','O')])
    # заполнить пустую сети CompositeModel
    for node in init.nodes: 
        parents = []
        for n in bn.nodes:
            if str(node) == str(n):
                parents = n.cont_parents + n.disc_parents
                break
        for n2 in init.nodes:
            if str(n2) in parents:
                node.nodes_from.append(n2)
    
# назначить parent_model узлам с родителями    
    for node in init.nodes:
        if not (node.nodes_from == None or node.nodes_from == []):
            node.content['parent_model'] = random_choice_model(node.content['type'])   

    
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
        # custom_mutation_add_structure, 
        # custom_mutation_delete_structure, 
        # custom_mutation_reverse_structure, 
        # custom_mutation_add_model,
        mutation_set1,
        mutation_set2,
        mutation_set3
        ],

        crossover_types = [custom_crossover_exchange_edges]
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
    n_generation = 50
    # вероятность кроссовера
    crossover_probability = 0.8
    # вероятность мутации
    mutation_probability = 0.9
    time_m = 40
    run_example() 