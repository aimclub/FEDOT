import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def synthetic_dataset(samples_amount, features_amount, classes_amount, full_shuffle=True):
    noisy_labels_fraction = 0.10
    features, target = datasets.make_classification(n_samples=samples_amount, n_features=features_amount,
                                                    n_informative=features_amount, n_redundant=0, n_repeated=0,
                                                    n_classes=classes_amount, n_clusters_per_class=1,
                                                    flip_y=noisy_labels_fraction,
                                                    shuffle=full_shuffle)

    return features, target


def gauss_quantiles(samples_amount, features_amount, classes_amount, full_shuffle=True):
    features, target = datasets.make_gaussian_quantiles(n_samples=samples_amount, n_features=features_amount,
                                                        n_classes=classes_amount, shuffle=full_shuffle)

    return features, target


if __name__ == '__main__':
    samples_total, features_amount = 100, 10
    classes = 2
    features, target = synthetic_dataset(samples_total, features_amount, classes, full_shuffle=False)

    plt.subplot(121)
    plt.title("Two informative features, one cluster per class")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    features, target = gauss_quantiles(samples_total, features_amount=2, classes_amount=classes)
    plt.subplot(122)
    plt.title("Gaussian divided into three quantiles")
    plt.scatter(features[:, 0], features[:, 1], marker='o', c=target,
                s=25, edgecolor='k')

    plt.show()
