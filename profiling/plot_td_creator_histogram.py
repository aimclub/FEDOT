"""
Build grouped bar charts from TensorDataCreator profiling CSV results.

Reads ``td_creator_profile_*.csv`` produced by
:mod:`profiling.td_creator_profiler` and writes per-backend PNG histograms
grouped by cache scenario and data size.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CASES = ("no_cache", "cache_write", "cache_read")
CASE_LABELS = {
    "no_cache": "No cache",
    "cache_write": "Cache write",
    "cache_read": "Cache read",
}
SIZE_LABELS = ("small", "medium", "large", "xlarge")

SIZE_COLORS = {
    "small": "#4C72B0",
    "medium": "#55A868",
    "large": "#C44E52",
    "xlarge": "#8172B3",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def find_latest_csv(results_dir: Path = RESULTS_DIR) -> Path:
    """
    Return the most recent TensorDataCreator profiling CSV in *results_dir*.

    Args:
        results_dir: Directory that stores ``td_creator_profile_*.csv`` files.

    Returns:
        Path to the latest CSV file sorted by filename.

    Raises:
        FileNotFoundError: If no matching CSV files exist.
    """
    files = sorted(results_dir.glob("td_creator_profile_*.csv"))
    if not files:
        raise FileNotFoundError(f"No td_creator_profile_*.csv files in {results_dir}")
    return files[-1]


def load_case_means(csv_path: Path, backend: str) -> dict[str, dict[str, float]]:
    """
    Return mean timing for each cache scenario and data size.

    Args:
        csv_path: TensorDataCreator profiling CSV from :func:`td_creator_profiler.run_profile`.
        backend: Backend label to filter rows by (for example ``cpu`` or ``cuda:2``).

    Returns:
        Nested mapping ``case -> size_label -> mean_s``.

    Raises:
        ValueError: If any expected case/size combination is missing.
    """
    values: dict[str, dict[str, list[float]]] = {
        case: {size: [] for size in SIZE_LABELS} for case in CASES
    }

    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["backend"] != backend:
                continue

            case = row["case"]
            size_label = row["size_label"]
            if case not in CASES or size_label not in SIZE_LABELS:
                continue

            values[case][size_label].append(float(row["mean_s"]))

    result: dict[str, dict[str, float]] = {}
    for case in CASES:
        result[case] = {}
        for size_label in SIZE_LABELS:
            samples = values[case][size_label]
            if not samples:
                raise ValueError(
                    f"Missing data for backend={backend!r}, "
                    f"case={case!r}, size={size_label!r} in {csv_path}"
                )
            result[case][size_label] = float(np.mean(samples))

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
    case_means: dict[str, dict[str, float]],
    *,
    backend: str,
    run_id: str,
    output_path: Path,
) -> None:
    """
    Render and save a grouped bar chart for one backend.

    Args:
        case_means: Mean timings grouped by cache scenario and size.
        backend: Backend label shown in the plot title.
        run_id: Profiling run identifier embedded in the output filename.
        output_path: Destination PNG path.
    """
    n_groups = len(CASES)
    n_sizes = len(SIZE_LABELS)
    group_width = 0.8
    bar_width = group_width / n_sizes

    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(11, 6))

    for size_idx, size_label in enumerate(SIZE_LABELS):
        heights = [case_means[case][size_label] for case in CASES]
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
    ax.set_xticklabels([CASE_LABELS[case] for case in CASES])
    ax.set_ylabel("Mean time, s (log scale)")
    ax.set_title(f"TensorDataCreator — {backend} (run {run_id})")
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
    Build TensorDataCreator histogram PNGs for every backend in a CSV file.

    Args:
        csv_path: Source CSV path. Defaults to the latest file in *output_dir*.
        output_dir: Directory where PNG files are written.

    Returns:
        Paths to the saved histogram images.
    """
    csv_path = csv_path or find_latest_csv(output_dir)
    run_id = csv_path.stem.removeprefix("td_creator_profile_")

    with csv_path.open(newline="", encoding="utf-8") as file:
        backends = sorted({row["backend"] for row in csv.DictReader(file)})

    saved: list[Path] = []
    for backend in backends:
        means = load_case_means(csv_path, backend)
        output_path = output_dir / f"td_creator_histogram_{_backend_output_name(backend)}_{run_id}.png"
        plot_histogram(means, backend=backend, run_id=run_id, output_path=output_path)
        saved.append(output_path)

    return saved


def main() -> None:
    """CLI entry point for TensorDataCreator histogram generation."""
    parser = argparse.ArgumentParser(
        description="Plot TensorDataCreator profiling histograms from CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to td_creator_profile_*.csv (default: latest in profiling/results)",
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
