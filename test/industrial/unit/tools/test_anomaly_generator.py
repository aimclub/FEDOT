import warnings

import matplotlib.pyplot as plt
import pytest
from matplotlib import get_backend

from fedot_ind.tools.synthetic.anomaly_generator import AnomalyGenerator


@pytest.fixture
def config():
    return {'dip': {'level': 20,
                    'number': 5,
                    'min_anomaly_length': 10,
                    'max_anomaly_length': 20},
            'peak': {'level': 2,
                     'number': 5,
                     'min_anomaly_length': 5,
                     'max_anomaly_length': 10},
            'decrease_dispersion': {'level': 70,
                                    'number': 2,
                                    'min_anomaly_length': 10,
                                    'max_anomaly_length': 15},
            'increase_dispersion': {'level': 50,
                                    'number': 2,
                                    'min_anomaly_length': 10,
                                    'max_anomaly_length': 15},
            'shift_trend_up': {'level': 10,
                               'number': 2,
                               'min_anomaly_length': 10,
                               'max_anomaly_length': 20},
            'add_noise': {'level': 80,
                          'number': 2,
                          'noise_type': 'uniform',
                          'min_anomaly_length': 10,
                          'max_anomaly_length': 20}
            }


@pytest.fixture
def synthetic_ts():
    return {'ts_type': 'sin',
            'length': 1000,
            'amplitude': 10,
            'period': 500}


def test_generate(config, synthetic_ts):
    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")

    generator = AnomalyGenerator(config=config)
    init_synth_ts, mod_synth_ts, synth_inters = generator.generate(
        time_series_data=synthetic_ts, plot=True, overlap=0.1)

    assert len(init_synth_ts) == len(mod_synth_ts)
    for anomaly_type in synth_inters:
        for interval in synth_inters[anomaly_type]:
            ts_range = range(len(init_synth_ts))
            assert interval[0] in ts_range and interval[1] in ts_range
