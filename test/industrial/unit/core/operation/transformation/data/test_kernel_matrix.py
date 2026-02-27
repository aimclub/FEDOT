
from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pytest
from scipy.spatial.distance import pdist

from fedot_ind.core.operation.transformation.data.kernel_matrix import TSTransformer, colorise
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


@pytest.fixture
def params():
    random_walk_config = {
        'ts_type': 'random_walk',
        'length': 500,
        'start_val': 36.6}
    ts = TimeSeriesGenerator(random_walk_config).get_ts()

    return dict(time_series=ts,
                rec_metric='cosine')


@pytest.fixture
def ts_transformer(params):
    return TSTransformer(**params)


def test_ts_to_recurrence_matrix(ts_transformer, params):
    matrix = ts_transformer.ts_to_recurrence_matrix()
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == params['time_series'].shape[0]


def test_ts_to_3d_recurrence_matrix(ts_transformer, params):
    matrix = ts_transformer.ts_to_3d_recurrence_matrix()
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == matrix.shape[2]
    assert matrix.shape[1] == params['time_series'].shape[0]


def test_colorise(ts_transformer, params):
    dist_matrix = pdist(metric=ts_transformer.rec_metric,
                        X=params['time_series'].reshape(-1, 1))
    color_matrix = colorise(dist_matrix)
    assert len(color_matrix.shape) == 1
    assert color_matrix.dtype == 'uint8'


def test_binarization(ts_transformer, params):
    dist_matrix = pdist(metric=ts_transformer.rec_metric,
                        X=params['time_series'].reshape(-1, 1))
    bin_matrix = ts_transformer.binarization(dist_matrix, threshold=None)

    assert len(bin_matrix.shape) == 1
    assert len(np.unique(bin_matrix)) <= 2


def test_get_recurrence_metrics(ts_transformer, params):
    matrix = ts_transformer.get_recurrence_metrics()
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == params['time_series'].shape[0]
