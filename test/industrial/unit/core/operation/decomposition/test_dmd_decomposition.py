import numpy as np

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.dmd_decomposition import \
    orthogonal_dmd_decompose, \
    rq, symmetric_decompose


def test_rq():
    A = np.random.rand(4, 3)
    R, Q = rq(A)
    assert np.allclose(np.dot(R, Q), A)


def test_orthogonal_dmd_decompose():
    X = np.random.rand(5, 5)
    Y = np.random.rand(5, 5)
    rank = 3
    A, eigen_vals, eigen_vectors = orthogonal_dmd_decompose(X, Y, rank)
    assert eigen_vals.shape == (3, 3)
    assert eigen_vectors.shape == (5, 3)


def test_symmetric_decompose():
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)
    rank = 2
    A = symmetric_decompose(X, Y, rank)
    assert A is not None
