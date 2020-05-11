import itertools
import random
from copy import copy
from random import uniform

import numpy as np

from core.models.data import InputData
from core.models.model import Model
from core.models.preprocessing import Normalization
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (
    chain_template_balanced_tree,
    show_chain_template,
    fit_template,
    real_chain
)
from experiments.generate_data import gauss_quantiles


def source_chain(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, with_gaussian=True, skip_fit=True)
    initialized_chain = real_chain(template)

    return initialized_chain


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

    mixed_target = np.expand_dims(mixed_target, axis=1)
    data_train = InputData(idx=np.arange(0, samples_total),
                           features=mixed_features, target=mixed_target,
                           task_type=MachineLearningTasksEnum.classification)

    mixed_dataset = copy(data_train)
    preprocessor = Normalization().fit(mixed_dataset.features)
    mixed_dataset.features = preprocessor.apply(mixed_dataset.features)

    return mixed_dataset, preprocessor


def _cluster_labels(clusters_amount, classes_amount=2):
    assert classes_amount == 2

    label_threshold = random.randint(1, clusters_amount - 1)

    labels = []
    for cluster_idx in range(clusters_amount):
        label = 0 if cluster_idx < label_threshold else 1
        labels.append(label)
    random.shuffle(labels)

    return labels


if __name__ == '__main__':
    data_fit, preprocessor = mixed_clusters_dataset(clusters=5, samples_total=10000, features_amount=2)
