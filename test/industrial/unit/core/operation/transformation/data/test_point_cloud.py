import pytest

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.data.kernel_matrix import TSTransformer
from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalTransformation


@pytest.fixture()
def basic_periodic_data():
    size = 300
    x0 = 1 * np.ones(size) + np.random.rand(size) * 1
    x1 = 3 * np.ones(size) + np.random.rand(size) * 2
    x2 = 5 * np.ones(size) + np.random.rand(size) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)
    return x


def test_TSTransformer(basic_periodic_data):
    transformer = TSTransformer(time_series=basic_periodic_data,
                                rec_metric="euclidean")
    result = transformer.get_recurrence_metrics()
    assert result.shape[0] > 0 and result.shape[1] > 0


def test_TopologicalTransformation_time_series_rolling_betti_ripser(
        basic_periodic_data):
    topological_transformer = TopologicalTransformation(
        time_series=basic_periodic_data,
        max_simplex_dim=1,
        epsilon=3,
        window_length=400)

    betti_sum = topological_transformer.time_series_rolling_betti_ripser(basic_periodic_data)

    assert len(betti_sum) != 0


def test_TopologicalTransformation_time_series_to_point_cloud(
        basic_periodic_data):
    topological_transformer = TopologicalTransformation(
        time_series=basic_periodic_data,
        max_simplex_dim=1,
        epsilon=3,
        window_length=400)
    assert len(topological_transformer.time_series_to_point_cloud(
        basic_periodic_data)) != 0
