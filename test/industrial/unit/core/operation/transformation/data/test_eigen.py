from fedot_ind.core.architecture.settings.computational import backend_methods as np

from fedot_ind.core.operation.transformation.data.eigen import calculate_corr_matrix, calculate_matrix_norms, \
    combine_eigenvectors, weighted_inner_product

SAMPLE_DATA = np.array([1, 2, 3, 4, 5])
TS_LENGTH = 5
WINDOW_LENGTH = 2
N_COMPONENTS = 15
TS_COMPS = np.random.rand(TS_LENGTH, N_COMPONENTS)


def test_weighted_inner_product():
    result = weighted_inner_product(
        SAMPLE_DATA, SAMPLE_DATA, WINDOW_LENGTH, TS_LENGTH)
    assert isinstance(result, float)


def test_calculate_matrix_norms():
    result = calculate_matrix_norms(TS_COMPS, WINDOW_LENGTH, TS_LENGTH)
    assert isinstance(result, np.ndarray)


def test_calculate_corr_matrix():
    result, components = calculate_corr_matrix(ts_comps=TS_COMPS,
                                               f_wnorms=calculate_matrix_norms(
                                                   TS_COMPS, WINDOW_LENGTH, TS_LENGTH),
                                               window_length=WINDOW_LENGTH,
                                               ts_length=TS_LENGTH)
    assert isinstance(result, np.ndarray)
    assert isinstance(components, list)
    assert np.max(result) <= 1
    assert np.min(result) >= 0


def test_combine_eigenvectors():
    result = combine_eigenvectors(TS_COMPS, WINDOW_LENGTH)
    assert isinstance(result, list)
