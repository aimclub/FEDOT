from typing import Dict

import numpy as np
from sklearn import datasets


def classification_dataset(samples_amount: int, features_amount: int, classes_amount: int,
                           features_options: Dict, noise_fraction: float = 0.1,
                           full_shuffle: bool = True, weights: list = None):
    """Generates a random dataset for ``n-class`` classification problem
    using scikit-learn API.

    Args:
        samples_amount: Total amount of samples in the resulted dataset.
        features_amount: Total amount of features per sample.
        classes_amount: The amount of classes in the dataset.
        features_options: The dictionary containing features options in key-value format

            .. details:: possible ``features_options`` variants:

                - ``informative`` -> the amount of informative features
                - ``redundant`` -> the amount of redundant features
                - ``repeated`` -> the amount of features that repeat the informative features
                - ``clusters_per_class`` -> the amount of clusters for each class

        noise_fraction: the fraction of noisy labels in the dataset
        full_shuffle: if true then all features and samples will be shuffled
        weights: The proportions of samples assigned to each class. If None, then classes are balanced

    Returns:
        array: features and target as numpy-arrays
    """

    features, target = datasets.make_classification(n_samples=samples_amount, n_features=features_amount,
                                                    n_informative=features_options['informative'],
                                                    n_redundant=features_options['redundant'],
                                                    n_repeated=features_options['repeated'],
                                                    n_classes=classes_amount,
                                                    n_clusters_per_class=features_options['clusters_per_class'],
                                                    weights=weights,
                                                    flip_y=noise_fraction,
                                                    shuffle=full_shuffle)

    return features, target


def regression_dataset(samples_amount: int, features_amount: int, features_options: Dict,
                       n_targets: int, noise: float = 0.0, shuffle: bool = True):
    """Generates a random dataset for regression problem using scikit-learn API.

    Args:
        samples_amount: total amount of samples in the resulted dataset
        features_amount: total amount of features per sample
        features_options: the dictionary containing features options in key-value format

            .. details:: possible ``features_options`` variants:

                - ``informative`` -> the amount of informative features
                - ``bias`` -> bias term in the underlying linear model

        n_targets: the amount of target variables
        noise: the standard deviation of the gaussian noise applied to the output
        shuffle: if ``True`` then all features and samples will be shuffled

    Returns:
        array: features and target as numpy-arrays
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
    """Generates a random dataset for n-class classification problem
    based on multi-dimensional gaussian distribution quantiles
    using scikit-learn API.

    Args:
        samples_amount: total amount of samples in the resulted dataset
        features_amount: total amount of features per sample
        classes_amount: the amount of classes in the dataset
        full_shuffle: if ``True`` then all features and samples will be shuffled
        kwargs: Optional['gauss_params'] mean and covariance values of the distribution

    Returns:
        array: features and target as numpy-arrays
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
    """The function generates a synthetic one-dimensional array without omissions

    Args:
        length: the length of the array
        periods: the number of periods in the sine wave

    Returns:
        array: an array without gaps
    """

    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise
    return synthetic_data
