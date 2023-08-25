from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.base_mutations import MutationStrengthEnum
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.verification import rules_by_task

#from fedot.core.pipelines.prediction_intervals.utils import pipeline_simple_structure


def get_ts_mutation(individual: Individual):
    """This function gets a mutation of a given Individual object.

    Args:
        individual: an individual to make mutations.

    Returns:
        Individual: a mutation of a given individual.
    """
    operations = ['arima', 'lagged', 'glm', 'ridge', 'sparse_lagged',
                  'lasso', 'ts_naive_average', 'locf', 'pca', 'linear', 'smoothing']

    parameters = GPAlgorithmParameters(mutation_strength=MutationStrengthEnum.strong, mutation_prob=1)
    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)
    rules = rules_by_task(task_type=TaskTypesEnum.ts_forecasting)
    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=rules,
                                                  task=Task(TaskTypesEnum.ts_forecasting))

    mutation = Mutation(parameters, requirements, graph_params)

    return mutation._mutation(individual)[0]


def get_mutations(individual: Individual, number_mutations: int):
    """For a given individaul this function obtains several its mutations.

    Args:
        individaul: an individual
        number_mutations: a required number mutations.

    Returns:
        list of mutations of given individual. Mutations can be identical.
    """
    mutations = [get_ts_mutation(individual) for _ in range(number_mutations)]

    return mutations


def get_different_mutations(individual: Individual, number_mutations: int):
    """For a given individaul this function obtains several different its mutations.

    Args:
        individaul: an individual
        number_mutations: a required number mutations.

    Returns:
        list of mutations of given individual. Mutations must be different.
    """
    mutations = []
    pipeline_list = []

    while len(mutations) < number_mutations:
        new_ind = get_ts_mutation(individual)
        if new_ind.graph not in pipeline_list:
            pipeline_list.append(new_ind.graph)
            mutations.append(new_ind)

    return mutations
