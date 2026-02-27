import pytest
import numpy as np
from fedot_ind.core.models.detection.probalistic.kalman import AbstractKalmanFilter, UnscentedKalmanFilter, reshape_z

# Fixture for creating an instance of AbstractKalmanFilter


@pytest.fixture
def abstract_kalman_filter():
    model_hyperparams = {}  # Replace with actual hyperparameters if needed
    return AbstractKalmanFilter(model_hyperparams)

# Fixture for creating an instance of UnscentedKalmanFilter


@pytest.fixture
def unscented_kalman_filter():
    model_hyperparams = {}  # Replace with actual hyperparameters if needed
    return UnscentedKalmanFilter(model_hyperparams)

# Test reshape_z function


def test_reshape_z():
    z = np.array([1, 2, 3])
    dim_z = 3
    ndim = 1
    reshaped_z = reshape_z(z, dim_z, ndim)
    assert reshaped_z.shape == (dim_z,)

    z = np.array([[1, 2, 3]])
    reshaped_z = reshape_z(z, dim_z, ndim)
    assert reshaped_z.shape == (dim_z,)

    z = np.array([[1], [2], [3]])
    reshaped_z = reshape_z(z, dim_z, ndim)
    assert reshaped_z.shape == (dim_z,)

# Test AbstractKalmanFilter initialization


def test_abstract_kalman_filter_initialization(abstract_kalman_filter):
    assert abstract_kalman_filter.state is None
    assert abstract_kalman_filter.control_transition_matrix is None
    assert abstract_kalman_filter.state_transition_matrix is None
    assert abstract_kalman_filter._mahalanobis is None

# Test UnscentedKalmanFilter unscented_transform method


def test_unscented_kalman_filter_unscented_transform(unscented_kalman_filter):
    sigmas = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Wm = np.array([0.5, 0.3, 0.2])
    Wc = np.array([0.5, 0.3, 0.2])
    x, P = unscented_kalman_filter.unscented_transform(sigmas, Wm, Wc)
    assert x is not None
    assert P is not None
