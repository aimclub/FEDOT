"""Compare the removed sklearn/host stratification with device-local Torch indices.

Run: python -m examples.benchmark.tensor_stratified_split --output /tmp/tensor_split.json
The legacy function exists only as the benchmark reference, never in core execution.
"""
import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from fedot.core.data.split.tensor_split import stratified_split_indices


def legacy_split_indices(target, train_size):
    # Previous eligibility check and sklearn call each copied labels to host.
    labels = target.detach().cpu().numpy()
    _, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2) or len(target) - train_size < len(counts):
        return None
    labels = target.detach().cpu().numpy()
    train, test = train_test_split(np.arange(len(target)), train_size=train_size,
                                  stratify=labels, random_state=42)
    return tuple(torch.as_tensor(indices, device=target.device) for indices in (train, test))


def measure(function):
    torch.cuda.synchronize()
    started = perf_counter()
    result = function()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1000, result


def run(output):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'classes': 5, 'warmups': 3, 'repeats': 9, 'measurements': []}
    generator = torch.Generator(device='cuda:0').manual_seed(17)
    order = random.Random(17)
    for size in [1000, 12000, 250000, 1000000]:
        target = torch.randint(5, (size,), generator=generator, device='cuda:0')
        train_size = int(size * .8)
        functions = {'sklearn_host': lambda: legacy_split_indices(target, train_size),
                     'torch_device': lambda: stratified_split_indices(target, train_size)}
        for function in functions.values():
            train, test = function()
            assert train.device == test.device == target.device
            assert len(train) == train_size
            assert torch.equal(torch.cat((train, test)).sort().values, torch.arange(size, device=target.device))
            counts = torch.bincount(target)
            assert torch.all((torch.bincount(target[train]) - counts * .8).abs() < 1)
            for _ in range(3):
                function()
        samples = {name: [] for name in functions}
        for _ in range(9):
            names = list(functions)
            order.shuffle(names)
            for name in names:
                elapsed, result = measure(functions[name])
                samples[name].append(elapsed)
                del result
        old_ms = statistics.median(samples['sklearn_host'])
        new_ms = statistics.median(samples['torch_device'])
        row = {'rows': size, 'sklearn_host_ms': old_ms, 'torch_device_ms': new_ms,
               'speedup': old_ms / new_ms, 'samples_ms': samples}
        report['measurements'].append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'samples_ms'}), flush=True)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
