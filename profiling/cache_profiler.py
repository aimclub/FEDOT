"""
Profile FEDOT disk-cache components on configurable data sizes.

The script benchmarks ``Hasher``, ``Saver``, ``Loader``, and ``Cacher`` on CPU and
CUDA backends, writes CSV/JSON summaries to ``profiling/results``, and builds
comparison histograms for each backend.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

import fedot.core.caching.cache_cleaner as cache_cleaner_module
import fedot.core.caching.index_db as index_db_module
import fedot.core.caching.inmemory_operations as inmemory_operations
import fedot.core.caching.normalization as normalization_module
import fedot.core.caching.tools as cache_tools
import fedot.core.caching.tracer as tracer_module
from fedot.core.backend.backend import Backend
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.cache_saver import Saver
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.index_db import CacheIndexDB
from fedot.core.caching.tools import ensure_cache_dirs
from fedot.core.data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.planner import PreprocessingPlan
from profiling.plot_cache_histogram import build_histograms
from profiling.time_counter import measure_wall_time, warmup_cuda

CUDA_DEVICE = "cuda:2"
REPEATS = 5
WARMUP_RUNS = 2

SIZES: dict[str, tuple[int, int]] = {
    "small": (1_000, 32),
    "medium": (10_000, 128),
    "large": (100_000, 128),
    "xlarge": (1_000_000, 128),
}

CACHE_DIR_MODULES = (
    index_db_module,
    inmemory_operations,
    cache_tools,
    cache_cleaner_module,
    normalization_module,
    tracer_module,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ProfileTarget:
    """Profiling target: CPU backend or GPU backend on a specific CUDA device."""

    backend_name: str
    cuda_device: str = CUDA_DEVICE

    @property
    def label(self) -> str:
        """Short backend label used in result files and summaries."""
        if self.backend_name == "cpu":
            return "cpu"
        return self.cuda_device

    @property
    def sync_cuda(self) -> bool:
        """Whether CUDA synchronization is required after each timed call."""
        return self.backend_name == "gpu"


@dataclass
class ProfileContext:
    """Runtime context shared by cache profiling sections."""

    target: ProfileTarget
    backend: Backend
    env: CacheEnvironment


@dataclass
class BenchStats:
    """Aggregated wall-clock statistics collected over repeated runs."""

    min_s: float
    max_s: float
    mean_s: float
    median_s: float
    times_s: list[float] = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        """Serialize timing statistics to a JSON-friendly mapping."""
        return {
            "min_s": self.min_s,
            "max_s": self.max_s,
            "mean_s": self.mean_s,
            "median_s": self.median_s,
            "runs": float(len(self.times_s)),
        }


@dataclass
class ProfileRow:
    """One benchmark result row written to CSV and JSON exports."""

    backend: str
    category: str
    operation: str
    size_label: str
    rows: int
    cols: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    file_size_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CacheEnvironment:
    """Redirect FEDOT cache paths to an isolated directory."""

    def __init__(self, root: Path):
        """
        Args:
            root: Temporary cache root patched into all FEDOT cache modules.
        """
        self.root = root
        for module in CACHE_DIR_MODULES:
            module.CACHE_DIR = root

    def clear(self) -> None:
        """Remove the temporary cache directory and recreate empty subfolders."""
        if self.root.exists():
            shutil.rmtree(self.root)
        ensure_cache_dirs()


def configure_backend(target: ProfileTarget) -> Backend:
    """
    Configure the active FEDOT backend for a profiling target.

    Args:
        target: CPU or CUDA profiling target.

    Returns:
        Initialized :class:`~fedot.core.backend.backend.Backend` instance.
    """
    backend = Backend()
    if target.backend_name == "cpu":
        backend.set("cpu")
        return backend

    backend.set("gpu")
    backend.device = torch.device(target.cuda_device)
    torch.cuda.set_device(target.cuda_device)
    warmup_cuda(device=target.cuda_device)
    return backend


def _sync_cuda(device: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def bench_wall(
    ctx: ProfileContext,
    fn: Callable[..., Any],
    *args: Any,
    repeats: int = REPEATS,
    warmup: int = WARMUP_RUNS,
    **kwargs: Any,
) -> BenchStats:
    """
    Repeatedly time ``fn`` and aggregate wall-clock statistics.

    Args:
        ctx: Profiling context with backend and CUDA sync settings.
        fn: Callable to benchmark.
        *args: Positional arguments forwarded to ``fn``.
        repeats: Number of measured runs.
        warmup: Number of untimed warmup runs.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Aggregated timing statistics for the measured runs.
    """
    sync_cuda = ctx.target.sync_cuda
    device = ctx.target.cuda_device

    for _ in range(warmup):
        fn(*args, **kwargs)
        if sync_cuda:
            _sync_cuda(device)

    times: list[float] = []
    for _ in range(repeats):
        result = measure_wall_time(
            fn,
            *args,
            sync_cuda=sync_cuda,
            device=device,
            **kwargs,
        )
        times.append(result.elapsed_s)

    return BenchStats(
        min_s=min(times),
        max_s=max(times),
        mean_s=statistics.mean(times),
        median_s=statistics.median(times),
        times_s=times,
    )


def make_raw_features(backend: Backend, rows: int, cols: int) -> Any:
    """
    Build a raw feature matrix on the active backend array module.

    Args:
        backend: Configured FEDOT backend.
        rows: Number of samples.
        cols: Number of features.

    Returns:
        ``(rows, cols)`` array suitable for ``Hasher.hash`` raw-input tests.
    """
    xp = backend.xp
    values = xp.arange(rows * cols, dtype=xp.float32)
    return values.reshape(rows, cols)


def make_tensor_data(backend: Backend, rows: int, cols: int) -> TensorData:
    """
    Build a minimal ``TensorData`` instance on the active backend device.

    Args:
        backend: Configured FEDOT backend.
        rows: Number of samples.
        cols: Number of features.

    Returns:
        Table classification ``TensorData`` used by saver/loader/cacher benches.
    """
    features = torch.arange(
        rows * cols,
        dtype=torch.float32,
        device=backend.device,
    ).reshape(rows, cols)
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
        features=features,
        target=torch.zeros(rows, dtype=torch.float32, device=backend.device),
        categorical_idx=[],
        numerical_idx=list(range(cols)),
    )


def _row(
    ctx: ProfileContext,
    category: str,
    operation: str,
    size_label: str,
    rows: int,
    cols: int,
    stats: BenchStats,
    *,
    file_size_bytes: int | None = None,
    **extra: Any,
) -> ProfileRow:
    """Create a normalized profiling row from timing statistics."""
    return ProfileRow(
        backend=ctx.target.label,
        category=category,
        operation=operation,
        size_label=size_label,
        rows=rows,
        cols=cols,
        mean_s=stats.mean_s,
        median_s=stats.median_s,
        min_s=stats.min_s,
        max_s=stats.max_s,
        file_size_bytes=file_size_bytes,
        extra=extra,
    )


def profile_hasher(ctx: ProfileContext) -> list[ProfileRow]:
    """Benchmark raw-feature and tensor-data hashing across all configured sizes."""
    rows_out: list[ProfileRow] = []

    for size_label, (rows, cols) in SIZES.items():
        raw_features = make_raw_features(ctx.backend, rows, cols)
        tensor_data = make_tensor_data(ctx.backend, rows, cols)

        raw_stats = bench_wall(ctx, Hasher.hash, raw_features)
        rows_out.append(
            _row(ctx, "hasher", "hash_raw_features", size_label, rows, cols, raw_stats)
        )

        td_stats = bench_wall(ctx, Hasher.hash, tensor_data)
        rows_out.append(
            _row(ctx, "hasher", "hash_tensor_data", size_label, rows, cols, td_stats)
        )

    return rows_out


def profile_saver(ctx: ProfileContext) -> list[ProfileRow]:
    """Benchmark ``Saver.save`` for tensor-data artifacts across all sizes."""
    rows_out: list[ProfileRow] = []

    for size_label, (rows, cols) in SIZES.items():
        tensor_data = make_tensor_data(ctx.backend, rows, cols)
        content_hash = Hasher.hash(tensor_data)

        def save_once() -> None:
            ctx.env.clear()
            response = Saver.save(tensor_data, content_hash)
            if not response.success:
                raise RuntimeError(f"Saver.save failed for {size_label}")

        stats = bench_wall(ctx, save_once)
        ctx.env.clear()
        response = Saver.save(tensor_data, content_hash)
        file_size = response.path.stat().st_size if response.path.exists() else None

        rows_out.append(
            _row(
                ctx,
                "saver",
                "save_tensor_data",
                size_label,
                rows,
                cols,
                stats,
                file_size_bytes=file_size,
            )
        )

    return rows_out


def profile_loader(ctx: ProfileContext) -> list[ProfileRow]:
    """Benchmark ``Loader.load`` for saved tensor-data artifacts across all sizes."""
    rows_out: list[ProfileRow] = []

    for size_label, (rows, cols) in SIZES.items():
        ctx.env.clear()
        tensor_data = make_tensor_data(ctx.backend, rows, cols)
        content_hash = Hasher.hash(tensor_data)
        save_response = Saver.save(tensor_data, content_hash)
        if not save_response.success:
            raise RuntimeError(f"Setup save failed for loader/{size_label}")

        path = str(save_response.path)
        file_size = save_response.path.stat().st_size

        stats = bench_wall(ctx, Loader.load, path, content_hash, "tensor_data")

        rows_out.append(
            _row(
                ctx,
                "loader",
                "load_tensor_data",
                size_label,
                rows,
                cols,
                stats,
                file_size_bytes=file_size,
            )
        )

    return rows_out


def profile_cacher(ctx: ProfileContext) -> list[ProfileRow]:
    """Benchmark ``Cacher.cache_tensor_data`` and ``load_tensor_data`` across sizes."""
    rows_out: list[ProfileRow] = []

    for size_label, (rows, cols) in SIZES.items():
        raw_features = make_raw_features(ctx.backend, rows, cols)
        tensor_data = make_tensor_data(ctx.backend, rows, cols)
        operation = PreprocessingPlan()

        def cache_once() -> None:
            ctx.env.clear()
            cacher = Cacher(index_db=CacheIndexDB(), use_cache=True)
            record = cacher.cache_tensor_data(
                output_data=tensor_data,
                input_data=raw_features,
                operation=operation,
                state="fit",
            )
            if record is None:
                raise RuntimeError(f"cache_tensor_data failed for {size_label}")

        cache_stats = bench_wall(ctx, cache_once)
        rows_out.append(
            _row(
                ctx,
                "cacher",
                "cache_tensor_data",
                size_label,
                rows,
                cols,
                cache_stats,
            )
        )

        ctx.env.clear()
        cacher = Cacher(index_db=CacheIndexDB(), use_cache=True)
        cacher.cache_tensor_data(
            output_data=tensor_data,
            input_data=raw_features,
            operation=operation,
            state="fit",
        )

        load_stats = bench_wall(
            ctx,
            cacher.load_tensor_data,
            input_data=raw_features,
            operation=operation,
        )
        rows_out.append(
            _row(
                ctx,
                "cacher",
                "load_tensor_data",
                size_label,
                rows,
                cols,
                load_stats,
            )
        )

    return rows_out


def run_sections(ctx: ProfileContext) -> dict[str, list[ProfileRow]]:
    """Run all cache component profiling sections for one backend context."""
    return {
        "hasher": profile_hasher(ctx),
        "saver": profile_saver(ctx),
        "loader": profile_loader(ctx),
        "cacher": profile_cacher(ctx),
    }


def write_csv(path: Path, rows: list[ProfileRow]) -> None:
    """Write flattened profiling rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "backend",
                "category",
                "operation",
                "size_label",
                "rows",
                "cols",
                "mean_s",
                "median_s",
                "min_s",
                "max_s",
                "file_size_bytes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.backend,
                    row.category,
                    row.operation,
                    row.size_label,
                    row.rows,
                    row.cols,
                    row.mean_s,
                    row.median_s,
                    row.min_s,
                    row.max_s,
                    row.file_size_bytes or "",
                ]
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a profiling payload to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _rows_to_section_payload(rows: list[ProfileRow]) -> list[dict[str, Any]]:
    """Convert profiling rows into the nested JSON section format."""
    return [
        {
            "backend": row.backend,
            "operation": row.operation,
            "size_label": row.size_label,
            "shape": [row.rows, row.cols],
            "timing": {
                "mean_s": row.mean_s,
                "median_s": row.median_s,
                "min_s": row.min_s,
                "max_s": row.max_s,
            },
            "file_size_bytes": row.file_size_bytes,
        }
        for row in rows
    ]


def run_profile(
    targets: list[ProfileTarget] | None = None,
    cuda_device: str = CUDA_DEVICE,
    repeats: int = REPEATS,
    warmup: int = WARMUP_RUNS,
) -> dict[str, Any]:
    """
    Run the full cache profiling suite and export CSV/JSON/histogram artifacts.

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
    all_rows: list[ProfileRow] = []
    sections_by_backend: dict[str, dict[str, list[ProfileRow]]] = {}

    for target in targets:
        backend = configure_backend(target)
        env = CacheEnvironment(RESULTS_DIR / f"cache_run_{run_id}" / target.label)
        ctx = ProfileContext(target=target, backend=backend, env=env)
        sections = run_sections(ctx)
        sections_by_backend[target.label] = sections
        all_rows.extend(row for section_rows in sections.values() for row in section_rows)
        env.clear()

    payload = {
        "run_id": run_id,
        "targets": [
            {"backend": target.backend_name, "device": target.label}
            for target in targets
        ],
        "repeats": repeats,
        "warmup_runs": warmup,
        "sizes": SIZES,
        "sections": {
            backend_label: {
                name: _rows_to_section_payload(section_rows)
                for name, section_rows in backend_sections.items()
            }
            for backend_label, backend_sections in sections_by_backend.items()
        },
    }

    write_csv(RESULTS_DIR / f"cache_profile_{run_id}.csv", all_rows)
    write_json(RESULTS_DIR / f"cache_profile_{run_id}.json", payload)

    csv_path = RESULTS_DIR / f"cache_profile_{run_id}.csv"
    histogram_paths = build_histograms(csv_path=csv_path, output_dir=RESULTS_DIR)
    payload["histograms"] = [str(path) for path in histogram_paths]

    return payload


def print_summary(payload: dict[str, Any]) -> None:
    """Print a human-readable summary of a cache profiling run."""
    print(f"Cache profiling finished: run_id={payload['run_id']}")
    for path in payload.get("histograms", []):
        print(f"Histogram: {path}")
    for target in payload["targets"]:
        backend_label = target["device"]
        print(f"\n=== {backend_label} ===")
        backend_sections = payload["sections"][backend_label]
        for section, rows in backend_sections.items():
            print(f"[{section}]")
            for row in rows:
                shape = "x".join(str(x) for x in row["shape"])
                print(
                    f"  {row['operation']:24} {row['size_label']:7} {shape:>16} "
                    f"median={row['timing']['median_s']:.6f}s"
                )


def main() -> None:
    """CLI entry point for cache profiling."""
    logging.getLogger().setLevel(logging.WARNING)
    payload = run_profile()
    print_summary(payload)


if __name__ == "__main__":
    main()
