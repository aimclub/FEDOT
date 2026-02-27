import logging
import time
import numpy as np
import torch
import pandas as pd

from fedot_ind.core.operation.transformation.representation.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceNumpy
from fedot_ind.core.operation.transformation.torch_backend.recurrence.recurrence_extractor import RecurrenceExtractor as RecurrenceTorch
from tests.unit.api.fixtures import warm_up_cuda_computations
from fedot_ind.tools.synthetic.ts_generator import SinWave
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def compare_recurrence(ts_name, ts_data):
    torch_data = torch.tensor(ts_data, dtype=torch.float32)
    # NumPy CPU
    rec_np = RecurrenceNumpy(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    start = time.perf_counter()
    rec_np.generate_recurrence_features(ts_data)
    t_np = time.perf_counter() - start

    # Torch CPU
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    start = time.perf_counter()
    rec_torch.generate_recurrence_features(torch_data)
    t_torch = time.perf_counter() - start

    # Torch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec_torch = RecurrenceTorch(params={"rec_metric": "cosine", "window_size": 50, 'stride': 2})
    warm_up_cuda_computations(device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    rec_torch.generate_recurrence_features(torch_data.to(device))
    end_event.record()
    torch.cuda.synchronize()
    t_torch_gpu = start_event.elapsed_time(end_event) / 1000

    return {
        "data": ts_name,
        "data_shape": ts_data.shape,
        "numpy_cpu_time": t_np,
        "torch_cpu_time": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "torch_gpu_time": t_torch_gpu,
        "speedup_gpu": round(t_np / t_torch_gpu, 2),
    }


def test_recurrence_features():
    """

    """
    length = 30000

    # generate synthetic data
    sin_ts = SinWave({"length": length}).get_only_sin_ts()
    sin_noise_ts = SinWave({"length": length}).get_ts()
    time_series = {"sin": sin_ts,
                   "sin+noise": sin_noise_ts}

    train_multi_np, _ = TimeSeriesDatasetsGenerator(num_samples=2,
                                                    max_ts_len=length,
                                                    binary=False,
                                                    test_size=0.1).generate_data()
    multi_np = np.array(train_multi_np[0].values)

    # batch of TS with one chanel
    results = []
    for name, data in time_series.items():
        logger.info(f"Start test for {name}.")
        result = compare_recurrence(name, data)
        results.append(result)
        assert result["speedup"] > 1, \
            f"CPU speedup is not positive for {name}"
        assert result["speedup_gpu"] > 1, \
            f"GPU speedup is not positive for {name}"
        assert result["speedup_gpu"] > result["speedup"], \
            f"GPU do not give more speedup than CPU for {name}"
        logger.info(f"Successful test for {name}.")

    # multi-channels
    result_batch = compare_recurrence("multichannel data", multi_np)
    results.append(result_batch)
    assert result_batch["speedup"] > 1, \
        "No speedup on CPU for multichannel data"
    assert result_batch["speedup_gpu"] > 1, \
        "No speedup on GPU for multichannel data on GPU"
    assert result_batch["speedup_gpu"] > result_batch["speedup"], \
        f"GPU do not give more speedup than CPU for multichannel data"
    logger.info("Successful test for multichannel data.")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = test_recurrence_features()
