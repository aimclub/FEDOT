import numpy as np
import pytest

from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


@pytest.fixture
def config():
    return dict(random_walk={'ts_type': 'random_walk',
                             'length': 1000,
                             'start_val': 36.6},
                sin={'ts_type': 'sin',
                     'length': 1000,
                     'amplitude': 10,
                     'period': 500},
                auto_regression={'ts_type': 'auto_regression',
                                 'length': 1000,
                                 'ar_params': [0.5, -0.3, 0.2],
                                 'initial_values': None},
                smooth_normal={'ts_type': 'smooth_normal',
                               'length': 1000,
                               'window_size': 300}

                )


@pytest.mark.parametrize('kind', ('random_walk', 'sin',
                         'auto_regression', 'smooth_normal'))
def test_get_ts(config, kind):
    specific_config = config[kind]
    generator = TimeSeriesGenerator(params=specific_config)
    ts = generator.get_ts()
    assert isinstance(ts, np.ndarray)
    assert len(ts) == specific_config['length']
