"""
pip install aeon
"""

# pragma: no cover
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference


def load_results(csv_path):
    """Load results from CSV file and prepare for critical difference plot.

    Args:
        csv_path: Path to the CSV file

    Returns:
        results_array: 2D array of shape (n_datasets, n_estimators)
        method_names: List of framework names
        dataset_names: List of dataset names
    """
    df = pd.read_csv(csv_path)

    # Extract dataset names and method names
    dataset_names = df["Task"].values
    method_names = df.columns[1:].tolist()

    # Convert the data to numpy array, handling missing values
    results = df.iloc[:, 1:].values

    # Replace '-' with NaN for proper numerical handling
    results = np.where(results == "-", np.nan, results)
    results = results.astype(float)

    # Shape should be (n_datasets, n_estimators) as expected by aeon
    results_array = results

    return results_array, method_names, dataset_names


def generate_cd_plot(csv_path, title, output_path=None, alpha=0.1, lower_better=False):
    """Generate a critical difference diagram from CSV results.

    Args:
        csv_path: Path to the CSV file with results
        title: Title for the plot
        output_path: Optional path to save the plot
        alpha: Significance level for statistical test (default: 0.1)
        lower_better: Whether lower scores are better (default: False for accuracy, True for error)
    """
    results_array, method_names, dataset_names = load_results(csv_path)

    print(f"\n{title}")
    print(f"Number of frameworks: {len(method_names)}")
    print(f"Number of datasets (total): {len(dataset_names)}")
    print(f"Frameworks: {method_names}")

    # Count missing values per framework
    print("\nMissing values per framework:")
    for i, method in enumerate(method_names):
        missing = np.isnan(results_array[:, i]).sum()
        print(f"  {method}: {missing}/{len(dataset_names)} datasets")

    # Remove datasets (rows) where ANY framework has missing values
    # This keeps all frameworks but only uses complete datasets
    valid_rows = ~np.any(np.isnan(results_array), axis=1)
    filtered_results = results_array[valid_rows]
    filtered_datasets = dataset_names[valid_rows]

    print(
        f"\nDatasets with complete results: {len(filtered_datasets)}/{len(dataset_names)}"
    )
    print(
        f"Removed {len(dataset_names) - len(filtered_datasets)} datasets with missing values"
    )

    # Create the critical difference plot
    fig, ax = plot_critical_difference(
        filtered_results, method_names, alpha=alpha, lower_better=lower_better
    )

    ax.set_title(title, fontsize=14, pad=20)

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    return fig, ax


def generate_combined_cd_plot(csv_paths, title, output_path=None, alpha=0.1):
    """Generate a single CD diagram from multiple CSV files (all datasets combined).

    For combining different metrics (ROC AUC, Log Loss, RMSE), we use ranks instead of raw scores.
    This allows fair comparison across different task types.

    Args:
        csv_paths: List of paths to CSV files
        title: Title for the plot
        output_path: Optional path to save the plot
        alpha: Significance level for statistical test (default: 0.1)
    """
    all_results = []
    all_dataset_names = []
    method_names = None

    for csv_path in csv_paths:
        results_array, methods, dataset_names = load_results(csv_path)

        if method_names is None:
            method_names = methods
        elif method_names != methods:
            raise ValueError(
                "All CSV files must have the same frameworks in the same order"
            )

        # Remove rows with any NaN values
        valid_rows = ~np.any(np.isnan(results_array), axis=1)
        filtered_results = results_array[valid_rows]
        filtered_datasets = dataset_names[valid_rows]

        all_results.append(filtered_results)
        all_dataset_names.extend(filtered_datasets)

    # Concatenate all results
    combined_results = np.vstack(all_results)

    print(f"\n{title}")
    print(f"Number of frameworks: {len(method_names)}")  # type: ignore
    print(f"Number of datasets (total): {len(all_dataset_names)}")
    print(f"Frameworks: {method_names}")

    # Count datasets per task type
    print("\nDatasets per task type:")
    for i, csv_path in enumerate(csv_paths):
        print(f"  {csv_path.stem}: {all_results[i].shape[0]} datasets")

    # Create the critical difference plot
    # Note: We don't specify lower_better because we're combining different metrics
    # The ranking will be done automatically for each dataset
    fig, ax = plot_critical_difference(
        combined_results,
        method_names,
        alpha=alpha,
        lower_better=False,  # Default, ranks are computed per dataset anyway
    )

    ax.set_title(title, fontsize=14, pad=20)

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    return fig, ax


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent / "all_frameworks_res"
    output_dir = Path(__file__).parent / "cd_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate CD plots for each task type
    print("=" * 60)
    print("Generating Critical Difference Diagrams")
    print("=" * 60)

    # Binary Classification (higher is better - ROC AUC)
    fig1, ax1 = generate_cd_plot(
        base_dir / "binary_classification.csv",
        "Binary Classification - ROC AUC Score",
        output_dir / "cd_binary_classification.png",
        alpha=0.1,
        lower_better=False,
    )

    # Multiclass Classification (lower is better - Log Loss)
    fig2, ax2 = generate_cd_plot(
        base_dir / "multiclass_classification.csv",
        "Multiclass Classification - Log Loss",
        output_dir / "cd_multiclass_classification.png",
        alpha=0.1,
        lower_better=True,
    )

    # Regression (lower is better - RMSE)
    fig3, ax3 = generate_cd_plot(
        base_dir / "regression.csv",
        "Regression - RMSE",
        output_dir / "cd_regression.png",
        alpha=0.1,
        lower_better=True,
    )

    # Combined CD plot for all datasets
    fig4, ax4 = generate_combined_cd_plot(
        [
            base_dir / "binary_classification.csv",
            base_dir / "multiclass_classification.csv",
            base_dir / "regression.csv",
        ],
        "All Datasets - Combined Performance",
        output_dir / "cd_all_datasets.png",
        alpha=0.1,
    )

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)

    # Show all plots
    plt.show()
