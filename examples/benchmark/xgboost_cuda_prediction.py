"""Compare FEDOT's XGBoost NumPy output path with CUDA prediction via DLPack.

Run from the repository root:
    python -m examples.benchmark.xgboost_cuda_prediction --output /tmp/xgboost_cuda_prediction.json

The benchmark covers regression, binary logistic and multiclass softprob.
It compares sklearn's host output with the GPU path in the FEDOT implementation.
"""
import argparse
import json
import platform
import random
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.models.boosting import (
    BoostingClassificationStrategy, BoostingRegressionStrategy,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def tensor_data(features, target=None, *, regression=False):
    width = features.shape[1]
    return TensorData(
        Task(TaskTypesEnum.regression if regression else TaskTypesEnum.classification),
        DataTypesEnum.tabular, features, target=target,
        idx=torch.arange(len(features), device=features.device), numerical_idx=list(range(width)),
        idx_mapping=dict(enumerate(range(width))), features_names=[f'f{i}' for i in range(width)],
    )


def time_call(function):
    torch.cuda.synchronize()
    started = perf_counter()
    result = function()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1000, result


def run(output, repeats, batch_sizes):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires a CUDA GPU')
    generator = torch.Generator(device='cuda:0').manual_seed(17)
    features = torch.randn(12000, 32, device='cuda:0', generator=generator)
    inference = torch.randn(max(batch_sizes), 32, device='cuda:0', generator=generator)
    noise = torch.randn(12000, device='cuda:0', generator=generator)
    targets = {
        'regression': features[:, 0] * 2 - features[:, 1] + noise * .1,
        'binary': (features[:, 0] + features[:, 1] * .3 + noise * .1 > 0).to(torch.int64),
        'multiclass': (features[:, :5] + features[:, 5:10] * .3).argmax(dim=1),
    }
    report = {
        'environment': {
            'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'xgboost': version('xgboost'), 'cupy': version('cupy-cuda12x'),
        },
        'setup': {'train_rows': 12000, 'features': 32, 'max_trees': 100, 'max_depth': 6,
                  'warmups': 3, 'repeats': repeats, 'batch_sizes': batch_sizes, 'seed': 17,
                  'use_eval_set': True, 'early_stopping_rounds': 15},
        'training': [], 'measurements': [],
    }
    order = random.Random(17)
    print(json.dumps(report['environment']), flush=True)
    for task_name, target in targets.items():
        regression = task_name == 'regression'
        operation = 'xgboostreg' if regression else 'xgboost'
        strategy_cls = BoostingRegressionStrategy if regression else BoostingClassificationStrategy
        params = OperationParameters.from_operation_type(
            operation, device='cuda:0', tree_method='hist', n_estimators=100, max_depth=6,
            random_state=17, nthread=2, n_jobs=2, enable_categorical=False,
            use_eval_set=True, early_stopping_rounds=15,
        )
        train = tensor_data(features, target, regression=regression)
        fit_ms, implementation = time_call(lambda: strategy_cls(operation, params).fit(train))
        assert implementation._prepare_features(train).device.type == 'cuda'
        assert implementation._prepare_target(train).device.type == 'cuda'
        report['training'].append({'task': task_name, 'fit_ms': fit_ms,
                                   'best_iteration': implementation.model.best_iteration,
                                   'trained_rounds': implementation.model.get_booster().num_boosted_rounds()})
        for batch_size in batch_sizes:
            data = tensor_data(inference[:batch_size], regression=regression)
            for mode in (['value'] if regression else ['probs', 'labels']):
                legacy_method = implementation.model.predict_proba if mode == 'probs' else implementation.model.predict
                native_method = implementation.predict_proba if mode == 'probs' else implementation.predict
                functions = {
                    'numpy_roundtrip': lambda: torch.as_tensor(
                        legacy_method(implementation._prepare_features(data)), device=data.device),
                    'cuda_dlpack': lambda: native_method(data),
                }
                reference, candidate = functions['numpy_roundtrip'](), functions['cuda_dlpack']()
                torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
                assert candidate.device == reference.device == data.device
                samples = {name: [] for name in functions}
                for _ in range(3):
                    for function in functions.values():
                        function()
                torch.cuda.synchronize()
                for _ in range(repeats):
                    names = list(functions)
                    order.shuffle(names)
                    for name in names:
                        elapsed, result = time_call(functions[name])
                        samples[name].append(elapsed)
                        del result
                old_ms = statistics.median(samples['numpy_roundtrip'])
                new_ms = statistics.median(samples['cuda_dlpack'])
                row = {
                    'task': task_name, 'mode': mode, 'batch_rows': batch_size,
                    'numpy_roundtrip_ms': old_ms, 'cuda_dlpack_ms': new_ms,
                    'speedup': old_ms / new_ms, 'saved_percent': (1 - new_ms / old_ms) * 100,
                    'result_shape': list(reference.shape), 'exact_match': True, 'samples_ms': samples,
                }
                report['measurements'].append(row)
                output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({key: value for key, value in row.items() if key != 'samples_ms'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[256, 10000, 250000])
    args = parser.parse_args()
    run(args.output, args.repeats, args.batch_sizes)
