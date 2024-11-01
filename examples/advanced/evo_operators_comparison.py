import os
from datetime import timedelta
from typing import Sequence, Optional

import numpy as np
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

mutation_labels = [
    'Mutation simple',
    'Mutation growth',
    'Mutation reduce',
    'Mutation all',
]

crossover_labels = [
    'Crossover one point',
    'Crossover subtree',
    'Crossover all',
]


def run_single(train_data,
               test_data,
               mutation_types,
               crossover_types,
               timeout: Optional[float] = 10,
               num_generations: int = 20,
               visualize: bool = False):
    task = Task(TaskTypesEnum.classification)
    ops = get_operations_for_task(task)
    requirements = PipelineComposerRequirements(
        primary=ops,
        secondary=ops,
        num_of_generations=num_generations,
        timeout=timedelta(minutes=timeout) if timeout else None,
        early_stopping_iterations=None,
        n_jobs=-1,
    )
    gp_params = GPAlgorithmParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=mutation_types,
        crossover_types=crossover_types,
    )

    composer = ComposerBuilder(task=Task(TaskTypesEnum.classification)). \
        with_metrics(ClassificationMetricsEnum.ROCAUC). \
        with_requirements(requirements). \
        with_optimizer_params(gp_params). \
        build()

    pipeline = composer.compose_pipeline(train_data)
    pipeline.fit_from_scratch(train_data)
    predicted = pipeline.predict(test_data)

    roc_auc_metric = roc_auc(y_true=test_data.target, y_score=predicted.predict)
    print('roc_auc=', roc_auc_metric)

    if visualize:
        pipeline.show()
        composer.history.show.fitness_line()

    return composer.history


def load_histories(history_dir, filename_filter=None):
    hs = []
    for obj in os.listdir(history_dir):
        fullpath = f'{history_dir}/{obj}'
        if not os.path.isfile(fullpath):
            continue
        if filename_filter and filename_filter not in str(obj):
            continue
        history = OptHistory.load(fullpath)
        hs.append(history)
    return hs


def visualize_histories(histories: Sequence[OptHistory],
                        labels: Sequence[str],
                        with_confidence_interval: bool = True,
                        ):
    best_num = 5
    for history, label in zip(histories, labels):
        h = history.historical_fitness[1:-1]  # without initial and last pop
        best_fitness = np.abs(np.array([np.min(pop) for pop in h]))

        ys = best_fitness
        xs = np.arange(0, len(best_fitness))
        plt.xticks(xs)
        plt.plot(xs, ys, label=label)

        if with_confidence_interval:
            best_num = min(len(xs), best_num)
            std_fitness = np.array([np.std(sorted(pop)[:best_num]) for pop in h])
            plt.fill_between(xs, ys + std_fitness, ys - std_fitness, alpha=0.2)

    plt.xlabel('Поколение')
    plt.ylabel('Метрика')
    plt.legend()
    plt.show()


def run_experiment(train_data_path,
                   test_data_path,
                   save_dir,
                   timeout_per_run: Optional[float] = 10,
                   num_generations: int = 20,
                   ):
    train_data = InputData.from_csv(train_data_path, target_columns='target')
    test_data = InputData.from_csv(test_data_path, target_columns='target')

    all_mutations = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
    mutation_types = [
        [MutationTypesEnum.simple],
        [MutationTypesEnum.growth],
        [MutationTypesEnum.reduce],
        all_mutations,
    ]
    all_crossovers = [CrossoverTypesEnum.one_point, CrossoverTypesEnum.subtree]
    crossover_types = [
        [CrossoverTypesEnum.one_point],
        [CrossoverTypesEnum.subtree],
        all_crossovers,
    ]

    mutation_histories = []
    for label, mutations in zip(mutation_labels, mutation_types):
        label = label.lower().replace(' ', '_')
        history_file_path = f'{save_dir}/{label}.json'

        history = run_single(train_data, test_data,
                             timeout=timeout_per_run,
                             num_generations=num_generations,
                             mutation_types=mutations,
                             crossover_types=all_crossovers)
        mutation_histories.append(history)
        print(f'history is saved to path {history_file_path}')
        history.save(history_file_path)

    crossover_histories = []
    for label, crossover in zip(crossover_labels, crossover_types):
        label = label.lower().replace(' ', '_')
        history_file_path = f'{save_dir}/{label}.json'

        history = run_single(train_data, test_data,
                             timeout=timeout_per_run,
                             mutation_types=all_mutations,
                             crossover_types=crossover)
        crossover_histories.append(history)
        print(f'history is saved to path {history_file_path}')
        history.save(history_file_path)

    visualize_histories(mutation_histories, mutation_labels)
    visualize_histories(crossover_histories, crossover_labels)


def run_experiment_with_saved_histories(save_dir):
    mutation_histories = load_histories(save_dir, 'mutation')
    visualize_histories(mutation_histories, mutation_labels)

    crossover_histories = load_histories(save_dir, 'crossover')
    visualize_histories(crossover_histories, crossover_labels)


if __name__ == '__main__':
    train_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/examples/real_cases/data/scoring/scoring_test.csv'

    run_experiment(train_data_path,
                   test_data_path,
                   save_dir='result_histories',
                   timeout_per_run=None,
                   num_generations=20,
                   )
