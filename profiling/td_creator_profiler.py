"""
Profile ``TensorDataCreator.create`` under different cache modes.

The script compares no-cache, cache-write, and cache-read scenarios on CPU and
CUDA backends, exports CSV/JSON summaries to ``profiling/results``, and builds
histograms for each backend.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from profiling.cache_profiler import (
    CUDA_DEVICE,
    REPEATS,
    RESULTS_DIR,
    WARMUP_RUNS,
    BenchStats,
    CacheEnvironment,
    ProfileContext,
    ProfileTarget,
    SIZES,
    bench_wall,
    configure_backend,
    make_raw_features,
)
from profiling.plot_td_creator_histogram import build_histograms
from profiling.time_counter import measure_wall_time

CASES = (
    "no_cache",
    "cache_write",
    "cache_read",
)

CASE_LABELS = {
    "no_cache": "No cache",
    "cache_write": "Cache write",
    "cache_read": "Cache read",
}


@dataclass
class TDCreatorRow:
    """One TensorDataCreator benchmark row written to CSV and JSON exports."""

    backend: str
    case: str
    size_label: str
    rows: int
    cols: int
    use_cache: bool
    mean_s: float
    median_s: float
    min_s: float
    max_s: float


def _copy_features(features: Any) -> Any:
    """Return a shallow copy of raw features for cache-read benchmarks."""
    return features.copy()


def _create(
    features: Any,
    backend_name: str,
    *,
    use_cache: bool,
) -> None:
    """Thin wrapper around :meth:`TensorDataCreator.create` for benchmarking."""
    TensorDataCreator.create(features, backend_name=backend_name, use_cache=use_cache)


def _sync_cuda(device: str) -> None:
    """Synchronize CUDA work on *device* when CUDA is available."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def bench_wall_with_setup(
    ctx: ProfileContext,
    fn: Any,
    setup: Any,
    repeats: int = REPEATS,
    warmup: int = WARMUP_RUNS,
) -> BenchStats:
    """
    Time ``fn`` after ``setup``; only ``fn`` is included in the measurement.

    Args:
        ctx: Profiling context with backend and CUDA sync settings.
        fn: Callable to benchmark.
        setup: Callable executed before each timed run.
        repeats: Number of measured runs.
        warmup: Number of warmup runs excluded from statistics.

    Returns:
        Aggregated timing statistics for the measured runs.
    """
    sync_cuda = ctx.target.sync_cuda
    device = ctx.target.cuda_device

    for _ in range(warmup):
        setup()
        fn()
        if sync_cuda:
            _sync_cuda(device)

    times: list[float] = []
    for _ in range(repeats):
        setup()
        result = measure_wall_time(fn, sync_cuda=sync_cuda, device=device)
        times.append(result.elapsed_s)

    return BenchStats(
        min_s=min(times),
        max_s=max(times),
        mean_s=statistics.mean(times),
        median_s=statistics.median(times),
        times_s=times,
    )


def _bench_case(
    ctx: ProfileContext,
    features: Any,
    case: str,
) -> BenchStats:
    """
    Benchmark one TensorDataCreator cache scenario.

    Args:
        ctx: Profiling context with isolated cache environment.
        features: Raw feature matrix passed to ``TensorDataCreator.create``.
        case: One of ``no_cache``, ``cache_write``, or ``cache_read``.

    Returns:
        Timing statistics for the selected scenario.

    Raises:
        ValueError: If *case* is not recognized.
    """
    backend_name = ctx.target.backend_name

    if case == "no_cache":
        return bench_wall(
            ctx,
            _create,
            features,
            backend_name,
            use_cache=False,
        )

    if case == "cache_write":
        def write_once() -> None:
            ctx.env.clear()
            _create(features, backend_name, use_cache=True)

        return bench_wall(ctx, write_once)

    if case == "cache_read":
        def warm_cache() -> None:
            ctx.env.clear()
            _create(features, backend_name, use_cache=True)

        def read_cached() -> None:
            _create(_copy_features(features), backend_name, use_cache=True)

        return bench_wall_with_setup(ctx, read_cached, warm_cache)

    raise ValueError(f"Unknown case: {case}")


def profile_td_creator(ctx: ProfileContext) -> list[TDCreatorRow]:
    """Benchmark all cache scenarios and sizes for one backend context."""
    rows: list[TDCreatorRow] = []

    for size_label, (rows_count, cols) in SIZES.items():
        features = make_raw_features(ctx.backend, rows_count, cols)

        for case in CASES:
            stats = _bench_case(ctx, features, case)
            rows.append(
                TDCreatorRow(
                    backend=ctx.target.label,
                    case=case,
                    size_label=size_label,
                    rows=rows_count,
                    cols=cols,
                    use_cache=case != "no_cache",
                    mean_s=stats.mean_s,
                    median_s=stats.median_s,
                    min_s=stats.min_s,
                    max_s=stats.max_s,
                )
            )

    return rows


def write_csv(path: Path, rows: list[TDCreatorRow]) -> None:
    """Write TensorDataCreator profiling rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "backend",
                "case",
                "use_cache",
                "size_label",
                "rows",
                "cols",
                "mean_s",
                "median_s",
                "min_s",
                "max_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.backend,
                    row.case,
                    row.use_cache,
                    row.size_label,
                    row.rows,
                    row.cols,
                    row.mean_s,
                    row.median_s,
                    row.min_s,
                    row.max_s,
                ]
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a TensorDataCreator profiling payload to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def run_profile(
    targets: list[ProfileTarget] | None = None,
    cuda_device: str = CUDA_DEVICE,
    repeats: int = REPEATS,
    warmup: int = WARMUP_RUNS,
) -> dict[str, Any]:
    """
    Run the TensorDataCreator profiling suite and export CSV/JSON/histograms.

    Args:
        targets: Backends to benchmark. Defaults to CPU and one CUDA device.
        cuda_device: CUDA device label used for the GPU target.
        repeats: Number of measured runs per benchmark.
        warmup: Number of warmup runs excluded from statistics.

    Returns:
        JSON-serializable profiling payload including output artifact paths.
    """
    global REPEATS, WARMUP_RUNS
    REPEATS = repeats
    WARMUP_RUNS = warmup

    if targets is None:
        targets = [
            ProfileTarget("cpu"),
            ProfileTarget("gpu", cuda_device),
        ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_rows: list[TDCreatorRow] = []

    for target in targets:
        backend = configure_backend(target)
        env = CacheEnvironment(RESULTS_DIR / f"td_creator_run_{run_id}" / target.label)
        ctx = ProfileContext(target=target, backend=backend, env=env)
        all_rows.extend(profile_td_creator(ctx))
        env.clear()

    payload = {
        "run_id": run_id,
        "targets": [
            {"backend": target.backend_name, "device": target.label}
            for target in targets
        ],
        "cases": {case: CASE_LABELS[case] for case in CASES},
        "repeats": repeats,
        "warmup_runs": warmup,
        "sizes": SIZES,
        "rows": [
            {
                "backend": row.backend,
                "case": row.case,
                "case_label": CASE_LABELS[row.case],
                "use_cache": row.use_cache,
                "size_label": row.size_label,
                "shape": [row.rows, row.cols],
                "timing": {
                    "mean_s": row.mean_s,
                    "median_s": row.median_s,
                    "min_s": row.min_s,
                    "max_s": row.max_s,
                },
            }
            for row in all_rows
        ],
    }

    csv_path = RESULTS_DIR / f"td_creator_profile_{run_id}.csv"
    write_csv(csv_path, all_rows)
    write_json(RESULTS_DIR / f"td_creator_profile_{run_id}.json", payload)

    histogram_paths = build_histograms(csv_path=csv_path, output_dir=RESULTS_DIR)
    payload["histograms"] = [str(path) for path in histogram_paths]

    return payload


def main() -> None:
    """CLI entry point for TensorDataCreator profiling."""
    logging.getLogger().setLevel(logging.WARNING)
    payload = run_profile()
    print(f"TensorDataCreator profiling finished: run_id={payload['run_id']}")
    for path in payload.get("histograms", []):
        print(f"Histogram: {path}")
    for target in payload["targets"]:
        backend_label = target["device"]
        print(f"\n=== {backend_label} ===")
        backend_rows = [row for row in payload["rows"] if row["backend"] == backend_label]
        for row in backend_rows:
            shape = "x".join(str(x) for x in row["shape"])
            print(
                f"  {row['case_label']:12} {row['size_label']:7} {shape:>16} "
                f"mean={row['timing']['mean_s']:.6f}s"
            )


if __name__ == "__main__":
    main()
