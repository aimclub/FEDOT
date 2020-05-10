import random
from random import uniform

import matplotlib.pyplot as plt
import numpy as np

from experiments.generate_data import gauss_quantiles


def mixed_clusters_dataset(clusters, samples_total, features_amount, classes=2):
    samples_per_cluster = samples_total // clusters
    mixed_features, mixed_target = [], []

    cluster_labels = _cluster_labels(clusters, classes)
    for cluster in range(clusters):
        mean = [uniform(-3, 3) for _ in range(features_amount)]
        cov = 1.
        features, target = gauss_quantiles(samples_per_cluster, features_amount, 1,
                                           gauss_params=(mean, cov))
        label = cluster_labels[cluster]
        target.fill(label)

        mixed_features.append(features)
        mixed_target.append(target)

    mixed_features = np.concatenate(mixed_features)
    mixed_target = np.concatenate(mixed_target)

    return mixed_features, mixed_target


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
    features, target = mixed_clusters_dataset(clusters=5, samples_total=10000, features_amount=2)
    plt.figure()
    plt.title("Test")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')
    plt.show()
