from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import (
    load_manifest,
    render_resolved_manifest,
    run_local_benchmark_preset,
    run_manifest_path,
    run_registered_manifest_path,
    run_registered_preset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run benchmark-v2 presets or manifest-driven jobs.')
    parser.add_argument('preset', nargs='?', choices=('m4', 'monash', 'ucr', 'tser'))
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--registered', action='store_true')
    parser.add_argument('--print-resolved-manifest', action='store_true')
    parser.add_argument('--dataset-name', default=None)
    parser.add_argument('--subset', default=None)
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--no-persist', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--include-optional-external', action='store_true')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.manifest:
        if args.print_resolved_manifest:
            resolved = render_resolved_manifest(load_manifest(args.manifest))
            print(json.dumps(resolved, indent=2, ensure_ascii=False))
            return 0
        if args.registered:
            bundle = run_registered_manifest_path(args.manifest)
            result = bundle.result
            print(f'Registry entry: {bundle.registry_entry_path}')
        else:
            result = run_manifest_path(args.manifest)
    else:
        if args.preset is None:
            parser.error('Either a preset positional argument or --manifest must be provided.')
        if args.registered:
            bundle = run_registered_preset(
                args.preset,
                dataset_name=args.dataset_name,
                subset=args.subset,
                sample_size=args.sample_size,
                output_dir=args.output_dir,
                persist_on_run=not args.no_persist,
                random_seed=args.random_seed,
                include_optional_external=args.include_optional_external,
            )
            result = bundle.result
            print(f'Registry entry: {bundle.registry_entry_path}')
        else:
            result = run_local_benchmark_preset(
                args.preset,
                dataset_name=args.dataset_name,
                subset=args.subset,
                sample_size=args.sample_size,
                output_dir=args.output_dir,
                persist_on_run=not args.no_persist,
                random_seed=args.random_seed,
                include_optional_external=args.include_optional_external,
            )

    successful_runs = sum(1 for record in result.run_records if record.status.value == 'success')
    output_dir = Path(result.config.artifact_spec.output_dir)
    if args.manifest:
        print(f'Manifest: {args.manifest}')
    else:
        print(f'Preset: {args.preset}')
    print(f'Run ID: {result.run_id}')
    print(f'Task: {result.config.task_type.value}')
    print(f'Successful runs: {successful_runs}')
    print(f'Primary metric: {result.aggregate_report.primary_metric}')
    print(f'Output dir: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
