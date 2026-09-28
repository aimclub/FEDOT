from typing import List, Tuple

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


def weighted_inner_product(
        F_i: np.ndarray,
        F_j: np.ndarray,
        window_length: int,
        ts_length: int) -> float:
    """Calculate the weighted inner product of two vectors.

    Args:
        F_i: First vector.
        F_j: Second vector.
        window_length: Length of the window.
        ts_length: Total length of the time series.

    Returns:
        Weighted inner product.
    """
    first = list(np.arange(window_length) + 1)
    second = [window_length] * (ts_length - 2 * window_length)
    third = list(np.arange(window_length) + 1)[::-1]
    w = np.array(first + second + third)
    return float(w.dot(F_i * F_j))


def calculate_matrix_norms(
        TS_comps: np.ndarray,
        window_length: int,
        ts_length: int) -> np.ndarray:
    """Calculate matrix norms for the time series components.

    Args:
        TS_comps: The time series components.
        window_length: Length of the window.
        ts_length: Total length of the time series.

    Returns:
        Array of matrix norms.
    """
    r = []
    for i in range(TS_comps.shape[1]):
        r.append(weighted_inner_product(
            TS_comps[:, i], TS_comps[:, i], window_length, ts_length))
    F_wnorms = np.array(r)
    F_wnorms = F_wnorms ** -0.5
    return F_wnorms


def calculate_corr_matrix(ts_comps: np.ndarray,
                          f_wnorms: np.ndarray,
                          window_length: int,
                          ts_length: int) -> Tuple[np.ndarray, List[int]]:
    """Calculate the w-correlation matrix for the time series components.

    Args:
        ts_comps: The time series components.
        f_wnorms: Matrix norms of the time series components.
        window_length: Length of the window.
        ts_length: Total length of the time series.

    Returns:
        W-correlation matrix and a list of component indices.
    """
    Wcorr = np.identity(ts_comps.shape[1])
    for i in range(Wcorr.shape[0]):
        for j in range(i + 1, Wcorr.shape[0]):
            Wcorr[i, j] = abs(
                weighted_inner_product(ts_comps[:, i], ts_comps[:, j], window_length, ts_length) *
                f_wnorms[i] * f_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]
    return Wcorr, [i for i in range(Wcorr.shape[0])]


def combine_eigenvectors(ts_comps: np.ndarray,
                         window_length: int) -> List[np.ndarray]:
    """Combine eigenvectors based on the w-correlation matrix for the time series.

    Args:
        ts_comps (np.ndarray): The time series components.
        window_length (int): Length of the window.

    Returns:
        List[np.ndarray]: List of combined eigenvectors.
    """
    ts_length = ts_comps.shape[0]
    F_wnorms = calculate_matrix_norms(ts_comps, window_length, ts_length)
    Wcorr, components = calculate_corr_matrix(
        ts_comps, F_wnorms, window_length, ts_length)
    combined_components = []
    current_group = []
    for i in range(len(components)):
        combined_components.append(np.array(current_group).sum(axis=0))
        current_group = [ts_comps[:, i]]
    combined_components.append(np.array(current_group).sum(axis=0))
    return combined_components
