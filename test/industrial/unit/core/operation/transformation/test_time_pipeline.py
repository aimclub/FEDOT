from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data_tensor, init_input_data
from fedot_ind.core.architecture.preprocessing.data_convertor import TensorConverter
from tests.unit.api.fixtures import warm_up_cuda_computations

import torch
import os
import shutil
import pandas as pd
import numpy as np
import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def remove_folder_completely(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def time_pipeline_test(DATASET_NAME="Beef"):
    cache_path = "/workspaces/Fedot.Industrial/cache"
    remove_folder_completely(cache_path)

    # Download data
    train_data, test_data = DataLoader(dataset_name=DATASET_NAME).load_data()

    # NumPy
    with IndustrialModels():
        pipeline_np = (
            PipelineBuilder()
            .add_node('quantile_extractor', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data_np = init_input_data(train_data[0], train_data[1])
        start_np = time.perf_counter()
        pipeline_np.fit(input_data_np)
        t_np = time.perf_counter() - start_np

    # Torch CPU
    converter = TensorConverter(data=train_data[0])
    with IndustrialModels():
        pipeline_torch = (
            PipelineBuilder()
            .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
            .add_node('rf')
            .build()
        )
        input_data_torch = init_input_data_tensor(converter.tensor_data, train_data[1])
        start_torch = time.perf_counter()
        pipeline_torch.fit(input_data_torch)
        t_torch = time.perf_counter() - start_torch

    remove_folder_completely(cache_path)

    # Torch GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        warm_up_cuda_computations(device=device)
        with IndustrialModels():
            pipeline_torch_gpu = (
                PipelineBuilder()
                .add_node('quantile_extractor_torch', params={'window_size': 20, 'window_mode': True})
                .add_node('rf')
                .build()
            )
            input_data_gpu = init_input_data_tensor(converter.tensor_data.to(device), train_data[1])
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pipeline_torch_gpu.fit(input_data_gpu)
            end_event.record()
            torch.cuda.synchronize()
            t_torch_gpu = start_event.elapsed_time(end_event) / 1000

    remove_folder_completely(cache_path)

    assert t_torch < t_np, "Torch CPU is not faster than NumPy CPU."
    assert t_torch_gpu < t_np, "Torch GPU is not faster than NumPy CPU."
    assert t_torch < t_torch_gpu, "Torch GPU is not faster than NumPy CPU."

    return {
        "dataset name": DATASET_NAME,
        "shape of data": input_data_np.features.shape,
        "numpy CPU time (sec)": t_np,
        "torch CPU time (sec)": t_torch,
        "speedup": round(t_np / t_torch, 2),
        "torch GPU time (sec)": t_torch_gpu,
        "speedup GPU": round(t_np / t_torch_gpu, 2) if device == "cuda" else np.nan,
    }


def run_pipeline_tests() -> pd.DataFrame:
    """Test to compare pipeline with Quantile Extractor"""
    datasets = ["WormsTwoClass"]  # "EthanolLevel", "UWaveGestureLibrary", "EMOPain"
    logger.info("Start test of pipeline.")
    results = []
    for dn in datasets:
        res = time_pipeline_test(dn)
        results.append(res)
    logger.info("Successful test.")
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_pipeline_tests()
