
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiment"
REPORTS = EXPERIMENT / "reports"

REPORTS.mkdir(exist_ok=True)

BENCHMARK_SRC = (
    ROOT.parent
    / "automl_multimodal_benchmark"
    / "multimodal_text_benchmark"
    / "src"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCHMARK_SRC))

from fedot.core.data.data_detection import TextDataDetector
from auto_mm_bench.datasets import dataset_registry


# ============================================================
# INPUT
# ============================================================

def find_predictions():
    paths = [
        REPORTS / "benchmark_predictions.csv",
        EXPERIMENT / "benchmark_predictions.csv",
    ]

    for path in paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Run benchmark_testing.py first"
    )


# ============================================================
# DETECTOR RULES
# ============================================================

def check_rules(features):
    """
    Reproduce the current improved detector's
    three natural-language detection rules.
    """

    f = features

    return {
        "long": {
            "median_word_count >= 7":
                f["median_word_count"] >= 7,

            "median_length >= 35":
                f["median_length"] >= 35,

            "alphabetic_ratio >= 0.55":
                f["alphabetic_ratio"] >= 0.55,
        },

        "medium": {
            "mean_word_count >= 5":
                f["mean_word_count"] >= 5,

            "mean_length >= 25":
                f["mean_length"] >= 25,

            "alphabetic_ratio >= 0.60":
                f["alphabetic_ratio"] >= 0.60,

            "digit_ratio < 0.25":
                f["digit_ratio"] < 0.25,

            "pattern_unique_ratio >= 0.15":
                f["pattern_unique_ratio"] >= 0.15,
        },

        "short": {
            "mean_word_count >= 4":
                f["mean_word_count"] >= 4,

            "mean_length >= 20":
                f["mean_length"] >= 20,

            "alphabetic_ratio >= 0.70":
                f["alphabetic_ratio"] >= 0.70,

            "digit_ratio < 0.15":
                f["digit_ratio"] < 0.15,

            "function_word_ratio >= 0.04":
                f["function_word_ratio"] >= 0.04,
        },
    }


def explain(features, detector):
    """
    Explain which rule rejected or accepted
    the column.
    """

    structured = [
        name
        for name in (
            "url_ratio",
            "email_ratio",
            "uuid_ratio",
            "path_ratio",
            "numeric_ratio",
            "id_ratio",
        )
        if features[name] >= detector.STRUCTURED_THRESHOLD
    ]

    if structured:
        return "STRUCTURED: " + ", ".join(structured)

    if features["unique_ratio"] < detector.MIN_UNIQUE_RATIO:
        return "LOW_UNIQUE_RATIO"

    rules = check_rules(features)

    passed = [
        name
        for name, checks in rules.items()
        if all(checks.values())
    ]

    if passed:
        return "TEXT_RULE: " + ", ".join(passed)

    # Find the rule with the fewest failed conditions
    closest = min(
        rules,
        key=lambda name: sum(
            not value
            for value in rules[name].values()
        ),
    )

    missing = [
        name
        for name, value in rules[closest].items()
        if not value
    ]

    return (
        f"NO_TEXT_RULE; closest={closest}; "
        f"missing: {', '.join(missing)}"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def main():

    predictions = pd.read_csv(find_predictions())

    # Only mistakes made by our production detector
    errors = predictions[
        (predictions["detector"] == "improved")
        & (
            predictions["expected"]
            != predictions["predicted"]
        )
    ]

    detector = TextDataDetector()

    results = []
    dataset_cache = {}

    for row in errors.itertuples(index=False):

        dataset_name = row.dataset
        column_name = row.column

        # Load each dataset once
        if dataset_name not in dataset_cache:
            dataset = dataset_registry.create(
                dataset_name,
                "train",
            )
            dataset_cache[dataset_name] = dataset.data

        dataframe = dataset_cache[dataset_name]
        column = dataframe[column_name]

        values = (
            column.dropna()
            .astype(str)
            .str.strip()
        )
        values = values[values != ""]

        if values.empty:
            continue

        # Extract the exact features used by FEDOT
        features = detector.extract_features(values)

        error_type = (
            "FN" if row.expected == 1 else "FP"
        )

        record = {
            "dataset": dataset_name,
            "column": column_name,
            "error": error_type,
            "expected": row.expected,
            "predicted": row.predicted,
            "reason": explain(features, detector),
            "example": (
                values.iloc[0]
                .replace("\n", " ")[:110]
            ),
        }

        important_features = [
            "rows",
            "unique_ratio",
            "mean_length",
            "median_length",
            "mean_word_count",
            "median_word_count",
            "alphabetic_ratio",
            "digit_ratio",
            "pattern_unique_ratio",
            "function_word_ratio",
            "tfidf_vocabulary_size",
            "url_ratio",
            "id_ratio",
        ]

        for feature in important_features:
            record[feature] = features[feature]

        results.append(record)

    if not results:
        print("No errors found for Improved detector")
        return

    report = pd.DataFrame(results)

    report = report.sort_values(
        ["error", "dataset", "column"]
    )

    output = REPORTS / "improved_errors_diagnostics.csv"

    report.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # HUMAN READABLE OUTPUT
    # ========================================================

    print("\n" + "=" * 65)
    print("IMPROVED DETECTOR ERROR ANALYSIS")
    print("=" * 65)

    print(f"\nTotal errors: {len(report)}")
    print(report["error"].value_counts())

    for row in report.itertuples(index=False):

        print("\n" + "-" * 65)

        print(f"{row.error}: {row.dataset} / {row.column}")

        print(f"Expected:  {row.expected}")
        print(f"Predicted: {row.predicted}")

        print(f"\nReason: {row.reason}")
        print(f"Example: {row.example}")

        print("\nStatistics:")
        print(f"  Mean words:     {row.mean_word_count:.2f}")
        print(f"  Median words:   {row.median_word_count:.2f}")
        print(f"  Mean length:    {row.mean_length:.2f}")
        print(f"  Unique ratio:   {row.unique_ratio:.3f}")
        print(f"  Alphabet ratio: {row.alphabetic_ratio:.3f}")
        print(f"  Function words: {row.function_word_ratio:.3f}")

    print("\n" + "=" * 65)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
