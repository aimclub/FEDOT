from abc import abstractmethod
from asyncio import CancelledError
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Sequence

from tqdm import tqdm

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import GraphFunction, ObjectiveFunction
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utilities.grouped_condition import GroupedCondition
from fpdf import FPDF
import numpy as np

if TYPE_CHECKING:
    pass

file = 'asia'
k = 2
dict_true_str = {'asia':
        [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')],

        'cancer':
        [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')],

        'earthquake':
        [('Burglary', 'Alarm'), ('Earthquake', 'Alarm'), ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')],

        'sachs':
        [('Erk', 'Akt'), ('Mek', 'Erk'), ('PIP3', 'PIP2'), ('PKA', 'Akt'), ('PKA', 'Erk'), ('PKA', 'Jnk'), ('PKA', 'Mek'), ('PKA', 'P38'), ('PKA', 'Raf'), ('PKC', 'Jnk'), ('PKC', 'Mek'), ('PKC', 'P38'), ('PKC', 'PKA'), ('PKC', 'Raf'), ('Plcg', 'PIP2'), ('Plcg', 'PIP3'), ('Raf', 'Mek')],  

        'healthcare':
        [('A', 'C'), ('A', 'D'), ('A', 'H'), ('A', 'O'), ('C', 'I'), ('D', 'I'), ('H', 'D'), ('I', 'T'), ('O', 'T')],
        
        'child':
        [('BirthAsphyxia', 'Disease'), ('HypDistrib', 'LowerBodyO2'), ('HypoxiaInO2', 'LowerBodyO2'), ('HypoxiaInO2', 'RUQO2'), ('CO2', 'CO2Report'), ('ChestXray', 'XrayReport'), ('Grunting', 'GruntingReport'), ('Disease', 'Age'), ('Disease', 'LVH'), ('Disease', 'DuctFlow'), ('Disease', 'CardiacMixing'), ('Disease', 'LungParench'), ('Disease', 'LungFlow'), ('Disease', 'Sick'), ('LVH', 'LVHreport'), ('DuctFlow', 'HypDistrib'), ('CardiacMixing', 'HypDistrib'), ('CardiacMixing', 'HypoxiaInO2'), ('LungParench', 'HypoxiaInO2'), ('LungParench', 'CO2'), ('LungParench', 'ChestXray'), ('LungParench', 'Grunting'), ('LungFlow', 'ChestXray'), ('Sick', 'Grunting'), ('Sick', 'Age')],
        
        'magic-niab':
        [('YR.GLASS', 'YR.FIELD'), ('YR.GLASS', 'YLD'), ('HT', 'YLD'), ('HT', 'FUS'), ('MIL', 'YR.GLASS'), ('FT', 'YR.FIELD'), ('FT', 'YLD'), ('G418', 'YR.GLASS'), ('G418', 'YR.FIELD'), ('G418', 'G1294'), ('G418', 'G2835'), ('G311', 'YR.GLASS'), ('G311', 'G43'), ('G1217', 'YR.GLASS'), ('G1217', 'MIL'), ('G1217', 'G257'), ('G1217', 'G1800'), ('G800', 'YR.GLASS'), ('G800', 'G383'), ('G866', 'YR.GLASS'), ('G795', 'YR.GLASS'), ('G2570', 'YLD'), ('G260', 'YLD'), ('G2920', 'YLD'), ('G832', 'HT'), ('G832', 'YLD'), ('G832', 'FUS'), ('G1896', 'HT'), ('G1896', 'FUS'), ('G2953', 'HT'), ('G2953', 'G1896'), ('G2953', 'G1800'), ('G266', 'HT'), ('G266', 'FT'), ('G266', 'G1789'), ('G847', 'HT'), ('G942', 'HT'), ('G200', 'YR.FIELD'), ('G257', 'YR.FIELD'), ('G257', 'G2208'), ('G257', 'G1800'), ('G2208', 'YR.FIELD'), ('G2208', 'MIL'), ('G1373', 'YR.FIELD'), ('G599', 'YR.FIELD'), ('G599', 'G1276'), ('G261', 'YR.FIELD'), ('G383', 'FUS'), ('G1853', 'G311'), ('G1853', 'FUS'), ('G1033', 'FUS'), ('G1945', 'MIL'), ('G1338', 'MIL'), ('G1338', 'G266'), ('G1276', 'FT'), ('G1276', 'G266'), ('G1263', 'FT'), ('G2318', 'FT'), ('G1294', 'FT'), ('G1800', 'FT'), ('G1750', 'YR.GLASS'), ('G1750', 'G1373'), ('G524', 'MIL'), ('G775', 'FT'), ('G2835', 'HT'), ('G2835', 'G1800')],
        
        'hack_processed_with_rf':
        [
        ('Tectonic regime','Structural setting'), ('Structural setting', 'Depth'), ('Structural setting', 'Gross'),
        ('Structural setting', 'Period'), ('Gross', 'Netpay'), ('Period', 'Porosity'),  ('Period', 'Gross'), 
        ('Porosity', 'Depth'), ('Porosity', 'Permeability'), ('Lithology', 'Gross'), ('Lithology', 'Permeability')
        ]
        }

def child_dict(net: list):
    res_dict = dict()
    for e0, e1 in net:
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict

def precision_recall(pred, true_net: list, decimal = 2):

    edges= pred.graph.operator.get_edges()
    struct = []
    for s in edges:
        struct.append((s[0].content['name'], s[1].content['name']))

    pred_net = deepcopy(struct)

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
    'SHD': shd}

true_net = dict_true_str[file]

class PopulationalOptimizer(GraphOptimizer):
    """
    Base class of populational optimizer.
    PopulationalOptimizer implements all basic methods for optimization not related to evolution process
    to experiment with other kinds of evolution optimization methods
    It allows to find the optimal solution using specified metric (one or several).
    To implement the specific evolution strategy,
    the abstract method '_evolution_process' should be re-defined in the ancestor class

    :param objective: objective for optimization
    :param initial_graphs: graphs which were initialized outside the optimizer
    :param requirements: implementation-independent requirements for graph optimizer
    :param graph_generation_params: parameters for new graph generation
    :param graph_optimizer_params: parameters for specific implementation of graph optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['GraphOptimizerParameters'] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.eval_dispatcher = MultiprocessingDispatcher(graph_adapter=graph_generation_params.adapter,
                                                         timer=self.timer,
                                                         n_jobs=requirements.n_jobs,
                                                         graph_cleanup_fn=_unfit_pipeline)

        # stopping_after_n_generation may be None, so use some obvious max number
        max_stagnation_length = requirements.early_stopping_generations or requirements.num_of_generations
        self.stop_optimization = \
            GroupedCondition().add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: self.current_generation_num >= requirements.num_of_generations + 1,
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: self.generations.stagnation_duration >= max_stagnation_length,
                'Optimisation finished: Early stopping criteria was satisfied'
            )

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_evaluation_callback(callback)

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective)

        with self.timer, self._progressbar:

            self._initial_population(evaluator=evaluator)
            
            # textfile = open("2_healthcare.txt", "w")
            # textfile.write('pop_size: ' + str(self.graph_optimizer_params.pop_size)+'\n')
            # textfile.write('num_of_generations: ' + str(self.requirements.num_of_generations)+'\n')
            # textfile.write('crossover_prob: ' + str(self.graph_optimizer_params.crossover_prob)+'\n')
            # textfile.write('mutation_prob: ' + str(self.graph_optimizer_params.mutation_prob)+'\n')
            # textfile.write('genetic_scheme_type: ' + str(self.graph_optimizer_params.genetic_scheme_type.name)+'\n')
            # textfile.write('selection_types: ' + str(self.graph_optimizer_params.selection_types[0].name)+'\n')
            # textfile.write('mutation_types: ' + str([i.__name__ for i in self.graph_optimizer_params.mutation_types])+'\n')
            # textfile.write('crossover_types: ' + str([i.__name__ for i in self.graph_optimizer_params.crossover_types])+'\n')
            # textfile.write('\n')        
            

            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(evaluator=evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    return self.best_graphs
                # Adding of new population to history
                self._update_population(new_population)
                # best = self.generations.best_individuals[0]
                # structure = best.graph.get_edges()
                # score = best.fitness.value[0]
                # SHD = precision_recall(best, true_net)['SHD']

                # textfile.write('current_generation_num:' + str(self.current_generation_num)+'\n')
                # textfile.write('structure:' + str(structure)+'\n')
                # textfile.write('score:' + str(score)+'\n')
                # textfile.write('SHD:' + str(SHD)+'\n')
                # textfile.write('\n')

        #         SHD = precision_recall(best, true_net)['SHD']
        #         print(SHD)
        #         OF = best.fitness.value[0]
        #         pdf.add_page()
        #         pdf.set_font("Arial", size = 14)
        #         pdf.cell(150, 5, txt = str(OF), ln = 1, align = 'C')   
        #         pdf.cell(150, 5, txt = str(SHD), ln = 1, align = 'C')   
        #         pdf.multi_cell(180, 5, txt = 'structure = ' + str(structure))  
        #         for node in best.graph.nodes:
        #             if node.content['parent_model'] == None: 
        #                 pdf.multi_cell(150, 5, txt = str(node) + " -> " + str(None))
        #             else:
        #                 pdf.multi_cell(150, 5, txt = str(node) + " -> " + str(node.content['parent_model'].implementation_info))
        # pdf.output("C:/Users/anaxa/Documents/Projects/CompositeBayesianNetworks/FEDOT/examples/pictures/asia0" +".pdf")
        
        
        # textfile.write('time_min:' + str(round(self.timer.minutes_from_start, 1)) + '\n')
        # textfile.write(str('parent_model:') + '\n')
        # for node in best.graph.nodes:
        #     if node.content['parent_model'] == None: 
        #         textfile.write(str(node) + " -> " + str(None) + '\n')
        #     else:
        #         textfile.write(str(node) + " -> " + str(node.content['parent_model'].implementation_info) + '\n')
                
        # textfile.close()

        best = self.generations.best_individuals[0]
        structure = best.graph.get_edges()
        score = best.fitness.value[0]
        SHD = precision_recall(best, true_net)['SHD']
        best.graph.show(save_path=('C:/Users/anaxa/Documents/Projects/CompositeBayesianNetworks/FEDOT/examples/pictures/train_test/' + str(k) + '_' + str(file) + '_' + str(self.number) + '.png'))

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 14)
        pdf.cell(150, 5, txt = 'Score = ' + str(score), ln = 1, align = 'C')   
        pdf.cell(150, 5, txt = 'SHD = ' + str(SHD), ln = 1, align = 'C')
        pdf.cell(150, 5, txt = 'time_min = ' + str(round(self.timer.minutes_from_start, 1)), ln = 1, align = 'C')
        pdf.cell(150, 5, txt = 'current_generation_num =' + str(self.current_generation_num), ln = 1, align = 'C')   
        pdf.image('C:/Users/anaxa/Documents/Projects/CompositeBayesianNetworks/FEDOT/examples/pictures/train_test/' + str(k) + '_' + str(file) + '_' + str(self.number) +'.png',w=165, h=165)     
        pdf.multi_cell(180, 5, txt = 'structure = ' + str(structure))  
        for node in best.graph.nodes:
            if node.content['parent_model'] == None: 
                pdf.multi_cell(150, 5, txt = str(node) + " -> " + str(None))
            else:
                pdf.multi_cell(150, 5, txt = str(node) + " -> " + str(node.content['parent_model'].implementation_info))
        pdf.output("C:/Users/anaxa/Documents/Projects/CompositeBayesianNetworks/FEDOT/examples/pictures/train_test/" + str(k) + '_' + str(file) + '_' + str(self.number) + 'train_test' +  ".pdf")
        

        return self.best_graphs
    @property
    def best_graphs(self):
        all_best_graphs = [ind.graph for ind in self.generations.best_individuals]
        return all_best_graphs

    @abstractmethod
    def _initial_population(self, *args, **kwargs):
        """ Initializes the initial population """
        raise NotImplementedError()

    @abstractmethod
    def _evolve_population(self, *args, **kwargs) -> PopulationT:
        """ Method realizing full evolution cycle """
        raise NotImplementedError()

    def _update_population(self, next_population: PopulationT):
        self._update_native_generation_numbers(next_population)
        self.generations.append(next_population)
        self._optimisation_callback(next_population, self.generations)
        self.population = next_population

        self.log.info(f'Generation num: {self.current_generation_num}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        self.log.info(f'no improvements for {self.generations.stagnation_duration} iterations')
        self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _update_native_generation_numbers(self, population: PopulationT):
        for individual in population:
            individual.set_native_generation(self.current_generation_num)

    @property
    def _progressbar(self):
        if self.requirements.show_progress:
            bar = tqdm(total=self.requirements.num_of_generations,
                       desc='Generations', unit='gen', initial=1)
        else:
            # disable call to tqdm.__init__ to avoid stdout/stderr access inside it
            # part of a workaround for https://github.com/nccr-itmo/FEDOT/issues/765
            bar = EmptyProgressBar()
        return bar


def _unfit_pipeline(graph: Any):
    if isinstance(graph, Pipeline):
        graph.unfit()


class EmptyProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class EvaluationAttemptsError(Exception):
    """ Number of evaluation attempts exceeded """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Too many fitness evaluation errors.'
