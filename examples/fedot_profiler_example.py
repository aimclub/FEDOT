import datetime
import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.chains.chain import Chain
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.data.data import InputData
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.utilities.profiler.profiler import MemoryProfiler, TimeProfiler

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    predicted = chain.predict(dataset_to_validate)
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target, y_score=predicted.predict)

    return roc_auc_value


def run_credit_scoring_problem(max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5)):
    train_file_path, test_file_path = get_scoring_case_data_paths()

    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    available_model_types = get_operations_for_task(task=task, mode='models')

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC_penalty)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=2, pop_size=2, num_of_generations=1,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time)

    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_optimiser_parameters(optimiser_parameters)

    composer = builder.build()
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose, is_visualise=True)
    chain_evo_composed.fit(input_data=dataset_to_compose)
    roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed, dataset_to_validate)

    print(f'Composed ROC AUC is {round(roc_on_valid_evo_composed, 3)}')

    return roc_on_valid_evo_composed


if __name__ == '__main__':
    # JUST UNCOMMENT WHAT TYPE OF PROFILER DO YOU NEED
    # EXAMPLE of MemoryProfiler.

    # path = os.path.join(os.path.expanduser("~"), 'memory_profiler')
    # profiler = MemoryProfiler(run_credit_scoring_problem, path=path, roots=run_credit_scoring_problem, max_depth=8)

    # EXAMPLE of TimeProfiler.

    profiler = TimeProfiler()
    run_credit_scoring_problem()
    path = os.path.join(os.path.expanduser("~"), 'time_profiler')
    profiler.profile(path=path, node_percent=0.5, edge_percent=0.1)
