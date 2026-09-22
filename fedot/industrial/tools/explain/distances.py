import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import energy_distance
from scipy.stats import entropy


def kl_divergence(probs_before: np.ndarray, probs_after: np.ndarray) -> float:
    """
    KL Divergence measures the information lost when one probability distribution is used to approximate another.

    Args:
        probs_before: The probability distribution before some event.
        probs_after: The probability distribution after the same event.

    """
    return entropy(probs_before, qk=probs_after)


def jensen_shannon_divergence(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence is a symmetric and smoothed version of KL Divergence, measuring
    the similarity between two distributions.

    Args:
        probs_before: The probability distribution before some event.
        probs_after: The probability distribution after the same event.

    """
    p = 0.5 * (probs_before + probs_after)
    return 0.5 * (entropy(probs_before, qk=p) + entropy(probs_after, qk=p))


def total_variation_distance(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Total Variation Distance measures the discrepancy between two probability distributions.
    It is also half the absolute area between the two curves

    Args:
        probs_before: The probability distribution before some event.
        probs_after: The probability distribution after the same event.

    """
    return 0.5 * np.sum(np.abs(probs_before - probs_after))


def energy_distance_measure(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Energy Distance measures the distance between the characteristic functions of two distributions.

    Args:
        probs_before: The probability distribution before some event.
        probs_after: The probability distribution after the same event.

    """
    return energy_distance(probs_before, probs_after)


def hellinger_distance(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Hellinger Distance measures the similarity between two probability distributions.

    Args:
        probs_before: The probability distribution before some event.
        probs_after: The probability distribution after the same event.

    """
    return np.sqrt(
        np.sum(
            (np.sqrt(probs_before) - np.sqrt(probs_after)) ** 2)) / np.sqrt(2)


def cosine_distance(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Cosine Distance measures the cosine of the angle between two vectors, indicating their similarity.

    Args:
        probs_before: A vector.
        probs_after: Another vector.

    """
    return cosine(probs_before, probs_after)


def euclidean_distance(
        probs_before: np.ndarray,
        probs_after: np.ndarray) -> float:
    """
    Euclidean Distance is the straight-line distance between two points in space.

    Args:
        probs_before: A vector.
        probs_after: Another vector.

    """
    return euclidean(probs_before, probs_after)


def cross_entropy(p, q):
    return -sum([p[i] * np.log2(q[i]) for i in range(len(p))])


def rmse(p, q):
    return np.sqrt(np.mean((p - q) ** 2))


DistanceTypes = dict(
    cosine=cosine_distance,
    euclidean=euclidean_distance,
    hellinger=hellinger_distance,
    energy=energy_distance_measure,
    total_variation=total_variation_distance,
    jensen_shannon=jensen_shannon_divergence,
    kl_div=kl_divergence,
    cross_entropy=cross_entropy,
    rmse=rmse
)
