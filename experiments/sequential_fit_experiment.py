from random import uniform

import matplotlib.pyplot as plt
import numpy as np

from experiments.generate_data import gauss_quantiles


def mixed_clusters_dataset(clusters, samples_total, features_amount, classes=2):
    samples_per_cluster = samples_total // clusters
    mixed_features, mixed_target = [], []
    for cluster in range(clusters):
        mean = [uniform(-3, 3) for _ in range(features_amount)]
        cov = 1.
        features, target = gauss_quantiles(samples_per_cluster, features_amount, classes,
                                           gauss_params=(mean, cov))

        mixed_features.append(features)
        mixed_target.append(target)

    mixed_features = np.concatenate(mixed_features)
    mixed_target = np.concatenate(mixed_target)

    return mixed_features, mixed_target


if __name__ == '__main__':
    features, target = mixed_clusters_dataset(clusters=4, samples_total=10000, features_amount=2)
    plt.figure()
    plt.title("Test")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')
    plt.show()
