from functools import partial
from typing import Tuple

import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import TaskTypesEnum, Task
from experiments.synth_generator.deap_example import run_evolution, chain_vs_single_model_fitness_diff, \
    individ_to_params, show_fitness_history
from experiments.synth_generator.fit_models import knn_score
from experiments.synth_generator.generators.mdc import generated_dataset
from experiments.synth_generator.mdc_gen_example import show_clusters


def default_mdc_dataset():
    params = {
        'n_samples': 2000,
        'n_feats': 2,
        'k': 3,
        'min_samples': 0,
        'possible_distributions': ['gaussian', 'gamma'],
        'corr': 0.,
        'compactness_factor': 0.1,
        'alpha_n': 1,
        'outliers': 50,
        'ki_coeff3': 3.
    }

    samples, labels = generated_dataset(params)

    return samples, labels


def fedot_input_data_format(mdc_dataset: Tuple[ndarray, ndarray]):
    samples, labels = mdc_dataset

    idx = np.arange(0, samples.shape[0])
    target = labels.reshape(-1)
    target += 1
    fedot_input = InputData(idx=idx, features=samples, target=target,
                            task=Task(TaskTypesEnum.classification),
                            data_type=DataTypesEnum.table)
    return fedot_input


def default_fedot_chain():
    first = PrimaryNode(model_type='logit')
    second = PrimaryNode(model_type='knn')
    third = SecondaryNode(model_type='xgboost', nodes_from=[first, second])

    chain = Chain()
    for node in [first, second, third]:
        chain.add_node(node)

    return chain


def accuracy(fitted_chain, data_true):
    predicted = fitted_chain.predict(data_true)
    predicted_classes = np.argmax(predicted.predict, axis=1)
    score = accuracy_score(data_true.target, predicted_classes)

    return score


def chain_score(dataset: Tuple, chain) -> Tuple[float, float]:
    samples, labels = dataset

    input_data = fedot_input_data_format(mdc_dataset=(samples, labels))
    data_compose, data_validate = train_test_data_setup(input_data)
    chain.fit(data_compose)
    print(f'Score on train: {accuracy(chain, data_compose)}')
    score = accuracy(fitted_chain=chain, data_true=data_validate)
    return score, 0.5


def chain_vs_single_eval_fitness(individual, chain):
    score = chain_vs_single_model_fitness_diff(individual,
                                               single_model_score=knn_score,
                                               chain_score=partial(chain_score, chain=chain))

    return score,


if __name__ == '__main__':
    chain = default_fedot_chain()
    top10, history = run_evolution(generations=10,
                                   fitness_eval=partial(chain_vs_single_eval_fitness, chain=chain))
    print(top10)
    best_params = top10[0]

    params_ = individ_to_params(best_params)
    params_['n_feat'] = 2
    samples, labels = generated_dataset(params=params_)
    show_clusters(samples=samples, labels=labels)
    show_fitness_history(history=history)
