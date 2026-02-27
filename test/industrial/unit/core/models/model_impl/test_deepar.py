import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from fedot_ind.core.models.nn.network_impl.forecasting_model.deepar import DeepAR, DeepARModule

FORECAST_LENGTH = 100


@pytest.fixture(scope='module')
def ts():
    features = np.random.rand(100)
    target = np.random.rand(100)
    task = Task(
        TaskTypesEnum.ts_forecasting,
        TsForecastingParams(
            forecast_length=FORECAST_LENGTH))
    inp_data = InputData(idx=np.arange(100),
                         features=features,
                         target=target,
                         task=task,
                         data_type=DataTypesEnum.ts)
    return inp_data


def test_prepare_data(ts):
    train_data = ts
    batch_size = 8
    patch_len = 13
    horizon = 2
    forecast_length = 11
    deepar = DeepAR({'forecast_length': forecast_length,
                     'patch_len': patch_len,
                     'batch_size': batch_size,
                     'horizon': horizon})
    train_batch_x, train_batch_y = next(
        iter(
            deepar._prepare_data(
                train_data, split_data=False, horizon=1)[0]))
    assert train_batch_x.ndim == 3, '3D output expected'
    assert train_batch_x.size(
        -1) == patch_len, 'Last dimension doesn\'t correspond patch_len'
    assert train_batch_x.size(
        0) == batch_size, 'First dimension doesn\'t correspond to batch_size'
    assert train_batch_y.ndim == 2, '2d output is expected for y'
    assert train_batch_y.size(-1) == 1, 'Last dim should be 1'

    test_batch_x, test_batch_y = next(iter(deepar._prepare_data(
        ts, split_data=False, horizon=horizon, is_train=True)[0]))
    assert test_batch_x.ndim == 3, 'Expected 3D output'
    assert test_batch_x.size(
        0) == batch_size, 'First dimension doesn\'t correspond to batch_size'
    assert test_batch_y.size(1) == horizon, 'Horizon expected to be different!'


@pytest.mark.skip('Problems with DeepAR dimensions based on output_mode')
def _test__predict(ts):
    deepar = DeepAR({'quantiles': [0.25, 0.5, 0.75]})
    deepar.fit(ts)

    n_samples = 100
    preds = deepar._predict(ts, output_mode='samples', n_samples=n_samples)[0]
    assert preds.ndim == 3, 'Dimensionality is not right'
    assert preds.size(-1) == n_samples, 'Predictions number is not correct'

    preds = deepar._predict(ts, output_mode='predictions')[0]
    assert preds.ndim == 2, 'Dimensionality is not right'
    assert preds.size(-1) == 1, 'Predictions should have 1 per index'
    assert preds.size(
        0) == deepar.forecast_length, 'forecast length doesn\'t correspond'

    preds = deepar._predict(ts, output_mode='raw')[0]
    p = len(deepar.loss_fn.distribution_arguments)
    assert preds.ndim == 3, 'Dimensionality is not right'
    assert preds.size(
        -1) == p, f'Predictions should have {p} per index for loss {type(deepar.loss_fn)}'

    preds = deepar._predict(ts, output_mode='quantiles')[0]
    q = len(deepar.model.quantiles)
    assert preds.ndim == 3, 'Dimensionality is not right'
    assert preds.size(
        -1) == q, f'Predictions should have {q} per index for quantiles range {deepar.quantiles}'


@pytest.mark.skip('Problems with DeepAR dimensions based on output_mode')
def _test_losses(ts):
    for loss_fn in DeepARModule._loss_fns:
        deepar = DeepAR({'expected_distribution': loss_fn})
        deepar.fit(ts)
        preds = deepar._predict(ts, output_mode='raw')
        p = len(deepar.loss_fn.distribution_arguments)
        assert preds.size(
            -1) == p, f'Predictions should have {p} per index for loss {loss_fn}'


@pytest.mark.parametrize('cell_type', [
    # 'RNN',
    'LSTM',
    'GRU'
])
def test_get_initial_state(ts, cell_type):
    deepar = DeepAR({'cell_type': cell_type})
    deepar.fit(ts)
    deepar.predict(ts)
