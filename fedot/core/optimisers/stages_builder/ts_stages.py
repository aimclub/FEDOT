from collections import defaultdict
from itertools import chain

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.optimisers.common_optimizer.nodes.evaluator import Evaluator
from golem.core.optimisers.common_optimizer.nodes.old_crossover import Crossover, CrossoverTask
from golem.core.optimisers.common_optimizer.nodes.old_elitism import Elitism, ElitismTask
from golem.core.optimisers.common_optimizer.nodes.old_inheritance import Inheritance, InheritanceTask
from golem.core.optimisers.common_optimizer.nodes.old_mutation import Mutation, MutationTask
from golem.core.optimisers.common_optimizer.nodes.old_regularization import Regularization, RegularizationTask
from golem.core.optimisers.common_optimizer.nodes.old_selection import Selection, SelectionTask
from golem.core.optimisers.common_optimizer.nodes.rebuild_population import PopulationRebuilder, PopulationRebuilderTask
from golem.core.optimisers.common_optimizer.nodes.uniqueness_check import UniquenessCheck, UniquenessCheckTask
from golem.core.optimisers.common_optimizer.runner import ParallelRunner, OneThreadRunner
from golem.core.optimisers.common_optimizer.scheme import Scheme, SequentialScheme
from golem.core.optimisers.common_optimizer.stage import Stage
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.core.optimisers.common_optimizer.temp.adaptive import AdaptivePopulationSize, AdaptiveParametersTask, \
    AdaptiveGraphDepth, AdaptiveOperatorsProb


ts_stages = list()


# adaptive parameters
nodes = [AdaptivePopulationSize(), AdaptiveGraphDepth(), AdaptiveOperatorsProb()]
scheme = SequentialScheme(nodes=nodes)
def adaptive_parameter_updater(finished_tasks, parameters):
    parameters = finished_tasks[0].update_parameters(parameters)
    return parameters

ts_stages.append(Stage(runner=OneThreadRunner(), nodes=nodes, task_builder=AdaptiveParametersTask,
                            scheme=scheme, stop_fun=lambda f, a: bool(f),
                            parameter_updater=adaptive_parameter_updater))

# main evolution process
class EvolvePopulationTask(MutationTask, CrossoverTask, RegularizationTask,
                           SelectionTask, UniquenessCheckTask, Task):
    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        parameters = super().update_parameters(parameters)
        return parameters

scheme_map = dict()
scheme_map[None] = defaultdict(lambda: 'regularization')
scheme_map['regularization'] = defaultdict(lambda: 'selection')
scheme_map['selection'] = defaultdict(lambda: 'crossover')
scheme_map['crossover'] = {TaskStatusEnum.SUCCESS: 'uniqueness_check_1', TaskStatusEnum.FAIL: None}
scheme_map['uniqueness_check_1'] = {TaskStatusEnum.SUCCESS: 'mutation', TaskStatusEnum.FAIL: None}
scheme_map['mutation'] = {TaskStatusEnum.SUCCESS: 'uniqueness_check_2', TaskStatusEnum.FAIL: None}
scheme_map['uniqueness_check_2'] = {TaskStatusEnum.SUCCESS: 'evaluator', TaskStatusEnum.FAIL: None}
scheme_map['evaluator'] = defaultdict(lambda: None)
scheme = Scheme(scheme_map=scheme_map)

nodes = [Mutation(), Crossover(), UniquenessCheck('uniqueness_check_1'), UniquenessCheck('uniqueness_check_2'),
         Regularization(), Selection(), Evaluator()]

def stop_fun(finished_tasks, all_tasks):
    if all_tasks:
        pop_size = all_tasks[0].graph_optimizer_params.pop_size
        if len(finished_tasks) >= pop_size or len(all_tasks) >= MAX_GRAPH_GEN_ATTEMPTS:
            return True
    return False

def parameter_updater(finished_tasks, parameters):
    new_population = chain(*[task.generation for task in finished_tasks])
    parameters.new_population = [ind for ind in new_population if ind.fitness.valid]
    return parameters

runner = ParallelRunner()
# runner = OneThreadRunner()
ts_stages.append(Stage(runner=runner, nodes=nodes, task_builder=EvolvePopulationTask,
                       scheme=scheme, stop_fun=stop_fun, parameter_updater=parameter_updater))


# inheritance, elitism, rebuild population
# may be done if population have been prepared
class FinalProcessingTask(ElitismTask, InheritanceTask, PopulationRebuilderTask, Task):
    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        parameters = super().update_parameters(parameters)
        return parameters

nodes = [Inheritance(), Elitism(), PopulationRebuilder()]
scheme = SequentialScheme(nodes=nodes)

def final_parameter_updater(finished_tasks, parameters):
    parameters = finished_tasks[0].update_parameters(parameters)
    return parameters

ts_stages.append(Stage(runner=OneThreadRunner(), nodes=nodes, task_builder=FinalProcessingTask,
                            scheme=scheme, stop_fun=lambda f, a: bool(f),
                            parameter_updater=final_parameter_updater))

