"""
Build grouped bar charts from cache profiling CSV results.

Reads ``cache_profile_*.csv`` produced by :mod:`profiling.cache_profiler` and
writes per-backend PNG histograms grouped by cache component and data size.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ("hasher", "saver", "loader")
SIZE_LABELS = ("small", "medium", "large", "xlarge")

OPERATIONS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "hasher": ("hash_raw_features", "hash_tensor_data"),
    "saver": ("save_tensor_data",),
    "loader": ("load_tensor_data",),
}

SIZE_COLORS = {
    "small": "#4C72B0",
    "medium": "#55A868",
    "large": "#C44E52",
    "xlarge": "#8172B3",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def find_latest_csv(results_dir: Path = RESULTS_DIR) -> Path:
    """
    Return the most recent cache profiling CSV in *results_dir*.

    Args:
        results_dir: Directory that stores ``cache_profile_*.csv`` files.

    Returns:
        Path to the latest CSV file sorted by filename.

    Raises:
        FileNotFoundError: If no matching CSV files exist.
    """
    files = sorted(results_dir.glob("cache_profile_*.csv"))
    if not files:
        raise FileNotFoundError(f"No cache_profile_*.csv files in {results_dir}")
    return files[-1]


def load_category_means(csv_path: Path, backend: str) -> dict[str, dict[str, float]]:
    """
    Return mean timing averaged over operations in each category and size.

    Args:
        csv_path: Cache profiling CSV produced by :func:`cache_profiler.run_profile`.
        backend: Backend label to filter rows by (for example ``cpu`` or ``cuda:2``).

    Returns:
        Nested mapping ``category -> size_label -> mean_s``.

    Raises:
        ValueError: If any expected category/size combination is missing.
    """
    values: dict[str, dict[str, list[float]]] = {
        category: {size: [] for size in SIZE_LABELS} for category in CATEGORIES
    }

    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["backend"] != backend:
                continue

            category = row["category"]
            operation = row["operation"]
            size_label = row["size_label"]
            if category not in OPERATIONS_BY_CATEGORY:
                continue
            if operation not in OPERATIONS_BY_CATEGORY[category]:
                continue
            if size_label not in SIZE_LABELS:
                continue

            values[category][size_label].append(float(row["mean_s"]))

    result: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        result[category] = {}
        for size_label in SIZE_LABELS:
            samples = values[category][size_label]
            if not samples:
                raise ValueError(
                    f"Missing data for backend={backend!r}, "
                    f"category={category!r}, size={size_label!r} in {csv_path}"
                )
            result[category][size_label] = float(np.mean(samples))

    return result


def _backend_output_name(backend: str) -> str:
    return "cpu" if backend == "cpu" else backend.replace(":", "_")


def _format_time(value: float) -> str:
    if value >= 0.01:
        return f"{value:.3f}s"
    if value >= 0.001:
        return f"{value * 1000:.1f}ms"
    return f"{value * 1000:.2f}ms"


def plot_histogram(
    category_means: dict[str, dict[str, float]],
    *,
    backend: str,
    run_id: str,
    output_path: Path,
) -> None:
    """
    Render and save a grouped bar chart for one backend.

    Args:
        category_means: Mean timings grouped by cache component and size.
        backend: Backend label shown in the plot title.
        run_id: Profiling run identifier embedded in the output filename.
        output_path: Destination PNG path.
    """
    n_groups = len(CATEGORIES)
    n_sizes = len(SIZE_LABELS)
    group_width = 0.8
    bar_width = group_width / n_sizes

    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(10, 6))

    for size_idx, size_label in enumerate(SIZE_LABELS):
        heights = [category_means[category][size_label] for category in CATEGORIES]
        offset = (size_idx - (n_sizes - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            label=size_label,
            color=SIZE_COLORS[size_label],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, heights):
            label_y = value * 1.08 if value > 0 else value
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                _format_time(value),
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in CATEGORIES])
    ax.set_ylabel("Mean time, s (log scale)")
    ax.set_title(f"Cache profiling — {backend} (run {run_id})")
    ax.legend(title="Size", loc="upper left")
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_histograms(
    csv_path: Path | None = None,
    output_dir: Path = RESULTS_DIR,
) -> list[Path]:
    """
    Build cache profiling histogram PNGs for every backend in a CSV file.

    Args:
        csv_path: Source CSV path. Defaults to the latest file in *output_dir*.
        output_dir: Directory where PNG files are written.

    Returns:
        Paths to the saved histogram images.
    """
    csv_path = csv_path or find_latest_csv(output_dir)
    run_id = csv_path.stem.removeprefix("cache_profile_")

    backends: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        backends = sorted({row["backend"] for row in csv.DictReader(file)})

    saved: list[Path] = []
    for backend in backends:
        means = load_category_means(csv_path, backend)
        output_path = output_dir / f"cache_histogram_{_backend_output_name(backend)}_{run_id}.png"
        plot_histogram(means, backend=backend, run_id=run_id, output_path=output_path)
        saved.append(output_path)

    return saved


def main() -> None:
    """CLI entry point for cache profiling histogram generation."""
    parser = argparse.ArgumentParser(description="Plot cache profiling histograms from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to cache_profile_*.csv (default: latest in profiling/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for PNG output",
    )
    args = parser.parse_args()

    saved = build_histograms(csv_path=args.csv, output_dir=args.output_dir)
    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
