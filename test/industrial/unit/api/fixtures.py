from functools import wraps
from threading import Thread

import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm

from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

NUM_SAMPLES = 50
SERIES_LENGTH = 20


@pytest.fixture
def univariate_time_series_np():
    return np.random.randn(NUM_SAMPLES, SERIES_LENGTH)


@pytest.fixture
def univariate_time_series_df():
    return pd.DataFrame(np.random.randn(NUM_SAMPLES, SERIES_LENGTH))


@pytest.fixture
def multivariate_time_series_np():
    return np.random.randn(NUM_SAMPLES, 3, SERIES_LENGTH)


@pytest.fixture
def multivariate_time_series_df():
    return pd.DataFrame(
        np.random.randn(
            NUM_SAMPLES,
            3,
            SERIES_LENGTH).tolist())


@pytest.fixture
def uni_classification_labels_np():
    return np.random.randint(0, 2, size=NUM_SAMPLES)


@pytest.fixture
def multi_classification_labels_np():
    return np.random.randint(0, 3, size=NUM_SAMPLES)


@pytest.fixture
def uni_classification_labels_df():
    return pd.DataFrame(np.random.randint(0, 2, size=NUM_SAMPLES))


@pytest.fixture
def multi_classification_labels_df():
    return pd.DataFrame(np.random.randint(0, 3, size=NUM_SAMPLES))


@pytest.fixture
def regression_target_np():
    return np.random.randn(NUM_SAMPLES)


@pytest.fixture
def regression_target_df():
    return pd.Series(np.random.randn(NUM_SAMPLES))


@pytest.fixture
def regression_multi_target_np():
    return np.random.randn(NUM_SAMPLES, 3)


@pytest.fixture
def regression_multi_target_df():
    return pd.DataFrame(np.random.randn(NUM_SAMPLES, 3))


def get_industrial_params():
    stat_params = {'window_size': 0, 'stride': 1, 'add_global_features': True,
                   'channel_independent': False, 'use_sliding_window': False}
    fourier_params = {'low_rank': 5, 'output_format': 'signal', 'compute_heuristic_representation': True,
                      'approximation': 'smooth', 'threshold': 0.9, 'sampling_rate': 64e3}
    wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7', 'compute_heuristic_representation': True}
    rocket_params = {'num_features': 200}
    sampling_dict = dict(samples=dict(start_idx=0, end_idx=None),
                         channels=dict(start_idx=0, end_idx=None),
                         elements=dict(start_idx=0, end_idx=None))
    feature_generator = {
        # 'minirocket': [('minirocket_extractor', rocket_params)],
        'stat_generator': [('quantile_extractor', stat_params)],
        'fourier': [('fourier_basis', fourier_params)],
        'wavelet': [('wavelet_basis', wavelet_params)],
    }

    return {'feature_generator': feature_generator,
            'data_type': 'tensor',
            'learning_strategy': 'all_classes',
            'head_model': 'rf',
            'sampling_strategy': sampling_dict}


def get_data_by_task(task: str, num_samples=None):
    if num_samples is None:
        num_samples = NUM_SAMPLES
    train_data, test_data = TimeSeriesDatasetsGenerator(num_samples=num_samples,
                                                        task=task,
                                                        max_ts_len=50,
                                                        binary=True,
                                                        test_size=0.5,
                                                        multivariate=False).generate_data()
    return train_data, test_data


def set_pytest_timeout_in_seconds(timeout_seconds):
    """
    A decorator to enforce a timeout on a test function.
    If the test exceeds the timeout, it will be skipped with a message including the parameterized input.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = dict(exception=None, value=None)

            def target():
                try:
                    result['value'] = func(*args, **kwargs)
                except Exception as e:
                    result['exception'] = e

            thread = Thread(target=target)
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                pytest.skip(f'Test skipped due to timeout after {timeout_seconds} seconds')
            if result['exception']:
                raise result['exception']
            return result['value']
        return wrapper
    return decorator


# Example usage in a test function:
def test_example(
        univariate_time_series_np,
        univariate_time_series_df,
        multivariate_time_series_np,
        multivariate_time_series_df,
        uni_classification_labels_np,
        multi_classification_labels_df,
        regression_target_np,
        regression_target_df):
    # Perform tests using the generated data
    assert len(univariate_time_series_np) == NUM_SAMPLES
    assert len(univariate_time_series_df) == NUM_SAMPLES
    assert len(multivariate_time_series_np) == NUM_SAMPLES
    assert len(multivariate_time_series_df) == NUM_SAMPLES
    assert len(uni_classification_labels_np) == NUM_SAMPLES
    assert len(multi_classification_labels_df) == NUM_SAMPLES
    assert len(regression_target_np) == NUM_SAMPLES
    assert len(regression_target_df) == NUM_SAMPLES


@torch.no_grad()
def warm_up_cuda_computations(n_iters=5, size=2048, device=None):
    """ Function for CUDA warming. It is used before time measuring.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    for _ in tqdm(range(n_iters)):
        C = A @ B
        C = torch.sin(C) * torch.exp(C)
        _ = C.sum()
    if device == "cuda":
        torch.cuda.synchronize()
