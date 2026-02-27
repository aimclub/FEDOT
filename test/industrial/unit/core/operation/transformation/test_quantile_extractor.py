import torch
import pandas as pd
import numpy as np
import time
import itertools
import logging
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot.core.operations.operation_parameters import OperationParameters
from tests.unit.api.fixtures import warm_up_cuda_computations
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


logger = logging.getLogger(__name__)


def test_stat_extractor(length=10000, window_size=10, stride=2, add_global_features=True):
    """ Function of measuring time of statictical features extracting for previous realisation (numpy)
    and new realisation (torch) on CPU and GPU. First, create random time series with shape shape_array,
    then extract features and measure time. Time for GPU is measured 10 times, then average value is taking.

    Returns:
        dict:
            length: length of time series in dataset.
            stride: stride for extractors
            add_global_features: parameter for extractors
            numpy CPU time (sec): time in seconds of extracting features through numpy realisation on CPU
            torch CPU time (sec): time in seconds of extracting features through torch realisation on CPU
            speedup: the value of deviding numpy CPU time by torch CPU time
            AVG torch GPU time (sec): average time in seconds of extracting features through torch realisation on GPU
            speedup GPU: the value of deviding numpy CPU time by AVG torch GPU time
            RMSE: root mean squared error between numpy and torch realisations
    """
    params = OperationParameters(
        window_size=window_size,
        stride=stride,
        add_global_features=add_global_features
    )
    numpy_extractor = QuantileExtractor(params)
    torch_extractor = TorchQuantileExtractor(params)

    # create synthetic data
    x_np, _ = TimeSeriesDatasetsGenerator(num_samples=2,
                                          max_ts_len=length,
                                          binary=False,
                                          test_size=0.1).generate_data()
    x_np = np.array(x_np[0].values).astype(np.float32)
    x_torch = torch.tensor(x_np, dtype=torch.float32)

    # numpy CPU
    start_np = time.perf_counter()
    np_result = numpy_extractor.generate_features_from_ts(x_np)
    t_np = time.perf_counter() - start_np

    # torch CPU
    start_torch = time.perf_counter()
    torch_result = torch_extractor.generate_features_from_ts(x_torch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_torch = time.perf_counter() - start_torch
    torch_result.detach().cpu().numpy()

    # torch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm_up_cuda_computations(device=device)
    times_gpu = []
    x_gpu = x_torch.clone().to(device)
    for i in range(10):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch_result_gpu = torch_extractor.generate_features_from_ts(x_gpu)
        end_event.record()
        torch.cuda.synchronize()
        t_torch_gpu = start_event.elapsed_time(end_event) / 1000
        times_gpu.append(t_torch_gpu)
    avg_time_gpu = np.mean(np.array(times_gpu))
    torch_result_gpu_np = torch_result_gpu.detach().cpu().numpy()
    rmse = np.power((np_result - torch_result_gpu_np), 2).mean() ** 0.5

    assert np_result.shape == torch_result_gpu_np.shape, "Shapes of numpy and torch results do not match"
    assert rmse < 1e-5, f"RMSE between numpy and torch results is too high: {rmse}"
    assert t_torch < t_np, "Torch CPU is not faster than NumPy CPU"
    assert avg_time_gpu < t_np, "Torch GPU is not faster than NumPy CPU"

    return {
        "length": length,
        "stride": stride,
        "add_global_features": add_global_features,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "AVG torch GPU time (sec)": avg_time_gpu,
        "speedup GPU": round(t_np / avg_time_gpu, 2),
        "RMSE": rmse,
    }


def run_stat_extractor_tests() -> pd.DataFrame:
    """
    Runs all combinations of tests for the statistical feature extractor.
    Returns a DataFrame with the results.
    """
    # parameters for test
    length = [10000, 30000]
    windows = [1, 10]
    strides = [1, 2]
    glob_feat = [True, False]

    logger.info(f"Start test of Quantile Extractor with different parameters.")
    results = []
    params_combinations = itertools.product(length, strides, glob_feat, windows)
    for len, stride, global_feat, w in params_combinations:
        res = test_stat_extractor(
            length=len,
            stride=stride,
            add_global_features=global_feat,
            window_size=w
        )
        results.append(res)
    logger.info(f"Successful test.")
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_stat_extractor_tests()
