from typing import Dict

from sklearn import datasets

import numpy as np


def classification_dataset(samples_amount: int, features_amount: int, classes_amount: int,
                           features_options: Dict, noise_fraction: float = 0.1,
                           full_shuffle: bool = True):
    """
    Generates a random dataset for n-class classification problem
    using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value \
        format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :param noise_fraction: the fraction of noisy labels in the dataset;
    :param full_shuffle: if true then all features and samples will be shuffled.
    :return: features and target as numpy-arrays.
    """
    features, target = datasets.make_classification(n_samples=samples_amount, n_features=features_amount,
                                                    n_informative=features_options['informative'],
                                                    n_redundant=features_options['redundant'],
                                                    n_repeated=features_options['repeated'],
                                                    n_classes=classes_amount,
                                                    n_clusters_per_class=features_options['clusters_per_class'],
                                                    flip_y=noise_fraction,
                                                    shuffle=full_shuffle)

    return features, target


def regression_dataset(samples_amount: int, features_amount: int, features_options: Dict,
                       n_targets: int, noise: float = 0.0, shuffle: bool = True):
    """
    Generates a random dataset for regression problem using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value \
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :param n_targets: the amount of target variables;
    :param noise: the standard deviation of the gaussian noise applied to the output;
    :param shuffle: if true then all features and samples will be shuffled.
    :return: features and target as numpy-arrays.
    """

    features, target = datasets.make_regression(n_samples=samples_amount, n_features=features_amount,
                                                n_informative=features_options['informative'],
                                                bias=features_options['bias'],
                                                n_targets=n_targets,
                                                noise=noise,
                                                shuffle=shuffle)

    return features, target


def gauss_quantiles_dataset(samples_amount: int, features_amount: int,
                            classes_amount: int, full_shuffle=True, **kwargs):
    """
    Generates a random dataset for n-class classification problem
    based on multi-dimensional gaussian distribution quantiles
    using scikit-learn API.

    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param full_shuffle: if true then all features and samples will be shuffled.
    :param kwargs: Optional params: \
        - 'gauss_params': mean and covariance values of the distribution.
    :return: features and target as numpy-arrays.
    """
    if 'gauss_params' in kwargs:
        mean, cov = kwargs['gauss_params']
    else:
        mean, cov = None, 1.

    features, target = datasets.make_gaussian_quantiles(n_samples=samples_amount,
                                                        n_features=features_amount,
                                                        n_classes=classes_amount,
                                                        shuffle=full_shuffle,
                                                        mean=mean, cov=cov)
    return features, target


def generate_synthetic_data(length: int = 2200, periods: int = 5):
    """
    The function generates a synthetic one-dimensional array without omissions

    :param length: the length of the array
    :param periods: the number of periods in the sine wave

    :return synthetic_data: an array without gaps
    """

    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise
    return synthetic_data
