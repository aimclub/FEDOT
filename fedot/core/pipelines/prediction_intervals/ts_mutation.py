from typing import List

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.base_mutations import MutationStrengthEnum
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.log import LoggerAdapter

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.verification import rules_by_task

from fedot.core.pipelines.prediction_intervals.graph_distance import get_distance_between


def get_ts_mutation(individual: Individual, operations: List[str]):
    """This function gets a mutation of a given Individual object.

    Args:
        individual: an individual to make mutations
        operations: list of possible mutations.

    Returns:
        Individual: a mutation of a given individual.
    """

    task_type = TaskTypesEnum.ts_forecasting
    parameters = GPAlgorithmParameters(mutation_strength=MutationStrengthEnum.strong, mutation_prob=1)
    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)
    rules = rules_by_task(task_type=task_type)
    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=rules,
                                                  task=Task(task_type))

    mutation = Mutation(parameters, requirements, graph_params)
    return mutation._mutation(individual)[0]


def get_mutations(individual: Individual, number_mutations: int, operations: List[str]):
    """For a given individaul this function obtains several its mutations.

    Args:
        individaul: an individual
        number_mutations: a required number mutations
        operations: list of possible mutations

    Returns:
        list of mutations of given individual. Mutations can be identical.
    """
    mutations = [get_ts_mutation(individual, operations) for _ in range(number_mutations)]
    return [x for x in mutations if x is not None]


def get_different_mutations(individual: Individual,
                            number_mutations: int,
                            operations: List[str],
                            logger: LoggerAdapter):
    """For a given individaul this function obtains several different its mutations.

    Args:
        individaul: an individual
        number_mutations: a required number mutations
        operations: list of possible mutations.

    Returns:
        list of mutations of given individual. Mutations must be different.
    """
    mutations, graph_list = [], []
    maximal_number_iterations = number_mutations * 3
    for _ in range(maximal_number_iterations):
        new_ind = get_ts_mutation(individual, operations)
        if new_ind is not None and all(get_distance_between(graph_1=new_ind.graph,
                                                            graph_2=x,
                                                            compare_node_params=False) for x in graph_list):
            graph_list.append(new_ind.graph)
            mutations.append(new_ind)
        if len(mutations) == number_mutations:
            break

    if len(mutations) != number_mutations:
        logger.warning(f"Maximal number attempts {maximal_number_iterations} to build different mutations used.")
    else:
        logger.info(f"{number_mutations} different mutations are succesfully created.")
    return mutations
