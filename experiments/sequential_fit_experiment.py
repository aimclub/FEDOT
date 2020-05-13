import itertools
import random
from collections import Counter
from copy import copy
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData, train_test_data_setup
from core.models.model import Model
from core.models.preprocessing import Normalization, DefaultStrategy
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (
    chain_template_balanced_tree,
    show_chain_template,
    fit_template,
    real_chain
)
from experiments.generate_data import gauss_quantiles


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.dt]
    return models


def source_chain_and_template(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, with_gaussian=True, skip_fit=True)
    initialized_chain = real_chain(template)

    return initialized_chain, template


def fit_model_templates(templates, data_fit, preprocessor):
    templates_by_models = []
    for model_template in itertools.chain.from_iterable(templates):
        model_instance = Model(model_type=model_template.model_type)
        model_template.model_instance = model_instance
        templates_by_models.append((model_template, model_instance))

    for template, instance in templates_by_models:
        print(f'Fit {instance}')
        fitted_model, predictions = instance.fit(data=data_fit)
        template.fitted_model = fitted_model
        template.data_fit = data_fit
        template.preprocessor = preprocessor


def mixed_clusters_dataset(clusters, samples_total, features_amount, classes=2):
    samples_per_cluster = samples_total // clusters
    mixed_features, mixed_target = [], []

    cluster_labels = _cluster_labels(clusters, classes)
    for cluster in range(clusters):
        mean = [uniform(-4, 4) for _ in range(features_amount)]
        cov = 1.
        features, target = gauss_quantiles(samples_per_cluster, features_amount, 1,
                                           gauss_params=(mean, cov))
        label = cluster_labels[cluster]
        target.fill(label)

        mixed_features.append(features)
        mixed_target.append(target)

    mixed_features = np.concatenate(mixed_features)
    mixed_target = np.concatenate(mixed_target)

    mixed_features, mixed_target = jointly_shuffled_values(mixed_features, mixed_target)
    mixed_target = np.expand_dims(mixed_target, axis=1)
    data_train = InputData(idx=np.arange(0, samples_total),
                           features=mixed_features, target=mixed_target,
                           task_type=MachineLearningTasksEnum.classification)

    mixed_dataset = copy(data_train)
    preprocessor = Normalization().fit(mixed_dataset.features)
    preprocessor = DefaultStrategy().fit(mixed_dataset.features)
    mixed_dataset.features = preprocessor.apply(mixed_dataset.features)

    return mixed_dataset, preprocessor


def jointly_shuffled_values(first, second):
    full = list(zip(first, second))
    random.shuffle(full)

    final_first, final_second = zip(*full)

    return final_first, final_second


def _cluster_labels(clusters_amount, classes_amount=2):
    assert classes_amount == 2

    label_threshold = random.randint(1, clusters_amount - 1)

    labels = []
    for cluster_idx in range(clusters_amount):
        label = 0 if cluster_idx < label_threshold else 1
        labels.append(label)
    random.shuffle(labels)

    return labels


def simple_chain_for_tests():
    samples, features, classes = 10000, 10, 2
    template = chain_template_balanced_tree(model_types=[ModelTypesIdsEnum.xgboost], depth=1, models_per_level=[1],
                                            samples=samples, features=features)
    show_chain_template(template)

    fit_template(template, classes=classes, with_gaussian=True, skip_fit=True)
    initialized_chain = real_chain(template)

    return initialized_chain


def roc_score(chain, data_to_compose, data_to_validate):
    predicted_train = chain.predict(data_to_compose)
    predicted_test = chain.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=data_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')

    return roc_train, roc_test


def mean_roc(runs=30):
    roc_train_full, roc_test_full = [], []
    for run in range(runs):
        chain = simple_chain_for_tests()
        data_fit, preprocessor = mixed_clusters_dataset(clusters=10, samples_total=10000, features_amount=2)

        plt.figure()
        plt.scatter(data_fit.features[:, 0], data_fit.features[:, 1], marker='o', c=data_fit.target[:, 0],
                    s=25, edgecolor='k')
        plt.savefig(f'mean_roc_figs/{run}.png')

        cnt = Counter([value for value in data_fit.target.flatten()])
        print(cnt)
        data_to_compose, data_to_validate = train_test_data_setup(data_fit)
        chain.fit_from_scratch(input_data=data_to_compose)
        roc_train, roc_test = roc_score(chain=chain, data_to_compose=data_to_compose,
                                        data_to_validate=data_to_validate)

        roc_train_full.append(roc_train)
        roc_test_full.append(roc_test)

    mean_train = np.mean(roc_train_full)
    std_train = np.std(roc_train_full)

    mean_test = np.mean(roc_test_full)
    std_test = np.std(roc_test_full)

    print(f'ROC Train: {mean_train} +/ {std_train}')
    print(f'ROC Test: {mean_test} +/ {std_test}')


if __name__ == '__main__':
    samples, features, classes = 10000, 10, 2
    # source_chain, template = source_chain_and_template(model_types=models_to_use(),
    #                                                    samples=samples, features=features, classes=classes)

    mean_roc(30)
