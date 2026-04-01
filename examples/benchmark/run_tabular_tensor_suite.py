from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fedot.industrial.benchmark.tabular_tensor_suite import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_MODES,
    build_config,
    run_tabular_tensor_suite,
)
from fedot.industrial.benchmark.tabular_tensor_suite.core import parse_csv_tuple  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run tabular TensorData benchmark suite.')
    parser.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    parser.add_argument('--modes', default=','.join(DEFAULT_MODES))
    parser.add_argument('--executor', default='sequential', choices=('sequential', 'dask_experimental'))
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--pop-size', type=int, default=15)
    parser.add_argument('--output-dir', default='benchmark_results/tabular_tensor_suite')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = build_config(
        datasets=parse_csv_tuple(args.datasets),
        modes=parse_csv_tuple(args.modes),
        executor=args.executor,
        seeds=parse_csv_tuple(args.seeds, cast=int),
        generations=args.generations,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
    )
    result = run_tabular_tensor_suite(config)
    print(f'Run ID: {result.run_id}')
    print(f'Artifacts: {", ".join(result.artifact_paths)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

