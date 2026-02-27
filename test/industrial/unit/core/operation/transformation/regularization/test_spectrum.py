import numpy as np
import pytest

from fedot_ind.core.operation.transformation.regularization.spectrum import reconstruct_basis, \
    singular_value_hard_threshold, sv_to_explained_variance_ratio, transform_eigen_to_ts, eigencorr_matrix
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


@pytest.fixture()
def matrix_from_ts():
    window = 30
    ts_config = {
        'ts_type': 'sin',
        'length': 300,
        'amplitude': 10,
        'period': 500
    }
    time_series = TimeSeriesGenerator(params=ts_config).get_ts()
    matrix = []
    for i in range(len(time_series) - window):
        matrix.append(time_series[i:i + window])
    return np.array(matrix)


@pytest.fixture()
def singular_values_rank_threshold_beta():
    return [531.37055486, 159.02909926, 121.41838726, 108.69788252,
            88.76097787, 59.24177627, 54.45026082, 39.17247824,
            26.70158246, 7.55227637], 3, 0.5, 0.5


def test_sv_to_explained_variance_ratio(singular_values_rank_threshold_beta):
    singular_values, rank, _, _ = singular_values_rank_threshold_beta
    explained_variance = sv_to_explained_variance_ratio(singular_values)
    assert 0 < sum(explained_variance) <= 100


def test_singular_value_hard_threshold(singular_values_rank_threshold_beta):
    singular_values, rank, beta, threshold = singular_values_rank_threshold_beta
    adjusted_sv = singular_value_hard_threshold(
        singular_values, rank, beta, threshold)
    assert len(adjusted_sv) == 3


def test_reconstruct_basis(matrix_from_ts):
    U, S, VT = np.linalg.svd(matrix_from_ts)
    reconstructed_basis = reconstruct_basis(U=U,
                                            Sigma=S,
                                            VT=VT,
                                            ts_length=299)
    assert isinstance(reconstructed_basis, np.ndarray)
    assert reconstructed_basis.shape == (299, 30)


@pytest.mark.parametrize('tensor, n_components', (
    (np.array([[1, 2, 2], [1, 3, 4], [3, 5, 6], [1, 4, 7]]), 2),
    (np.array([[1, 2, 2], [1, 3, 4], [3, 5, 6], [1, 4, 7]]), None),
))
def test_eigencorr_matrix(tensor, n_components):
    Ut, St, Vt = np.linalg.svd(tensor, full_matrices=False)
    corellated_components = eigencorr_matrix(U=Ut, S=St, V=Vt, n_components=n_components)
    assert isinstance(corellated_components, dict)
    print(corellated_components)


@pytest.mark.parametrize(
    'input_matrix, expected_output', [
        (np.array([[1, 2], [3, 4]]), [1.0, 2.5, 4.0]),
        (np.array([[5, 6], [7, 8]]), [5.0, 6.5, 8.0]),
        (np.array([[1, 0], [0, 1]]), [1.0, 0.0, 1.0]),
        (np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]), [0.0, 1.0, 2.0, 3, 4.0]),
    ]
)
def test_transform_eigen_to_ts(input_matrix, expected_output):
    result = transform_eigen_to_ts(input_matrix)
    assert result == expected_output
