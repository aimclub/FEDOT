
import argparse
import sys
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiment"

BENCHMARK_SRC = (
    ROOT.parent
    / "automl_multimodal_benchmark"
    / "multimodal_text_benchmark"
    / "src"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(BENCHMARK_SRC))


from fedot.core.data.data_detection import TextDataDetector
from text_detector_v2 import TextDataDetectorV2

from auto_mm_bench.datasets import (
    dataset_registry,
    TEXT_BENCHMARK_ALIAS_MAPPING,
)


LABELS = EXPERIMENT / "benchmark_columns_to_label.csv"


# ============================================================
# CURRENT FEDOT BASELINE
# ============================================================

class CurrentFedotDetector:
    """
    Reconstructed historical FEDOT text detector.
    """

    @staticmethod
    def _float_compatible(column: pd.Series) -> bool:

        values = column.dropna()

        if values.empty:
            return False

        converted = pd.to_numeric(
            values,
            errors="coerce",
        )

        return converted.isna().mean() < 0.5

    def define_text_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        detected = []

        for name in dataframe.columns:

            column = dataframe[name]

            if column.dtype != object:
                continue

            if self._float_compatible(column):
                continue

            values = column.dropna()

            if values.empty:
                continue

            unique_ratio = (
                values.nunique()
                / len(values)
            )

            if unique_ratio > 0.95:
                detected.append(name)

        return detected


# ============================================================
# PR #903 BASELINE
# ============================================================

class Pr903Detector:
    """
    Experimental reproduction of the
    text detection approach proposed in PR #903.
    """

    def define_text_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        detected = []

        for name in dataframe.columns:

            column = dataframe[name]

            if column.dtype != object:
                continue

            if CurrentFedotDetector._float_compatible(
                column
            ):
                continue

            values = (
                column.dropna()
                .astype(str)
            )

            if values.empty:
                continue

            unique_ratio = (
                values.nunique()
                / len(values)
            )

            if unique_ratio <= 0.60:
                continue

            link_ratio = values.str.match(
                r"^https?://",
                case=False,
            ).mean()

            if link_ratio > 0.5:
                continue

            try:
                vectorizer = TfidfVectorizer()
                vectorizer.fit(values)

            except ValueError:
                continue

            vocabulary_size = len(
                vectorizer.vocabulary_
            )

            if vocabulary_size > 20:
                detected.append(name)

        return detected


# ============================================================
# LABELS
# ============================================================

def load_manual_labels() -> dict:

    if not LABELS.exists():
        raise FileNotFoundError(
            f"Labels not found: {LABELS}"
        )

    dataframe = pd.read_csv(
        LABELS,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "dataset",
        "column",
        "target",
    }

    if not required.issubset(dataframe.columns):
        raise ValueError(
            f"Missing columns: "
            f"{required - set(dataframe.columns)}"
        )

    labels = {}

    for row in dataframe.itertuples(index=False):

        value = str(row.target).strip()

        if value not in {"0", "1"}:
            continue

        key = (
            str(row.dataset).strip(),
            str(row.column),
        )

        if key in labels:
            raise ValueError(
                f"Duplicate label: {key}"
            )

        labels[key] = int(value)

    return labels


def get_expected_labels(
    dataset_name,
    dataset,
    manual_labels,
):

    columns = list(
        dataset.feature_columns
    )

    try:
        feature_types = list(
            dataset.feature_types
        )

    except (
        AttributeError,
        NotImplementedError,
    ):
        feature_types = None

    if (
        feature_types is not None
        and len(feature_types) == len(columns)
    ):
        return {
            name: int(feature_type == "text")
            for name, feature_type in zip(
                columns,
                feature_types,
            )
        }, []

    expected = {}
    missing = []

    for column in columns:

        key = (
            dataset_name,
            column,
        )

        if key in manual_labels:
            expected[column] = manual_labels[key]
        else:
            missing.append(column)

    return expected, missing


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict:

    y_true = dataframe["expected"].astype(int)
    y_pred = dataframe["predicted"].astype(int)

    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    return {
        "columns": len(dataframe),

        "text_columns": int(
            y_true.sum()
        ),

        "tp": tp,
        "fp": fp,
        "fn": fn,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if (
        args.max_rows is not None
        and args.max_rows < 3
    ):
        parser.error(
            "--max-rows must be at least 3"
        )

    # ================================================
    # OUTPUT DIRECTORY
    # ================================================

    output_dir = EXPERIMENT / "reports"

    if args.split != "train":
        output_dir = output_dir / args.split

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================
    # LOAD DETECTORS
    # ================================================

    detectors = {
        "current_fedot": CurrentFedotDetector(),
        "pr903": Pr903Detector(),
        "improved_v1": TextDataDetector(),
        "improved_v2": TextDataDetectorV2(),
    }

    manual_labels = load_manual_labels()

    predictions = []
    diagnostics = []
    failed = []
    missing_labels = []

    dataset_names = list(
        TEXT_BENCHMARK_ALIAS_MAPPING.values()
    )

    # ================================================
    # EVALUATE DATASETS
    # ================================================

    for dataset_name in dataset_names:

        print()
        print("=" * 70)
        print(f"Processing: {dataset_name}")

        try:
            dataset = dataset_registry.create(
                dataset_name,
                args.split,
            )

            columns = list(
                dataset.feature_columns
            )

            expected, missing = get_expected_labels(
                dataset_name,
                dataset,
                manual_labels,
            )

            if missing:

                print(
                    f"SKIPPED: {len(missing)} missing labels"
                )

                for column in missing:
                    missing_labels.append({
                        "dataset": dataset_name,
                        "column": column,
                    })

                continue

            dataframe = dataset.data[columns]

            if args.max_rows is not None:
                dataframe = dataframe.head(
                    args.max_rows
                )

            # ========================================
            # COMPARE DETECTORS
            # ========================================

            for detector_name, detector in detectors.items():

                if detector_name == "improved_v2":

                    predicted_columns = set()

                    for column in columns:

                        analysis = detector.analyze_column(
                            dataframe[column]
                        )

                        if analysis["is_text"]:
                            predicted_columns.add(column)

                        diagnostic = {
                            "dataset": dataset_name,
                            "column": column,

                            "reason": analysis["reason"],

                            "expected": expected[column],

                            "predicted": int(
                                analysis["is_text"]
                            ),
                        }

                        diagnostic.update({
                            f"feature_{key}": value

                            for key, value
                            in analysis["features"].items()
                        })

                        diagnostics.append(diagnostic)

                else:
                    predicted_columns = set(
                        detector.define_text_columns(
                            dataframe
                        )
                    )

                # ====================================
                # SAVE COLUMN PREDICTIONS
                # ====================================

                for column in columns:

                    target = expected[column]

                    prediction = int(
                        column in predicted_columns
                    )

                    if target == prediction:

                        error_type = (
                            "TP" if target == 1
                            else "TN"
                        )

                    else:

                        error_type = (
                            "FN" if target == 1
                            else "FP"
                        )

                    predictions.append({
                        "dataset": dataset_name,
                        "column": column,
                        "detector": detector_name,

                        "expected": target,
                        "predicted": prediction,

                        "correct": int(
                            target == prediction
                        ),

                        "error_type": error_type,
                    })

        except Exception as error:

            print(f"FAILED: {error}")

            failed.append({
                "dataset": dataset_name,
                "error": repr(error),
            })

    # ================================================
    # SAVE FAILED / MISSING DATASETS
    # ================================================

    if missing_labels:

        pd.DataFrame(
            missing_labels
        ).to_csv(
            output_dir / "missing_labels.csv",
            index=False,
        )

    if failed:

        pd.DataFrame(
            failed
        ).to_csv(
            output_dir / "failed_datasets.csv",
            index=False,
        )

    if not predictions:
        raise RuntimeError(
            "No datasets were evaluated"
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    # ================================================
    # SUMMARY METRICS
    # ================================================

    summary_rows = []

    for detector_name, group in predictions_df.groupby(
        "detector",
        sort=False,
    ):

        metrics = calculate_metrics(
            group
        )

        summary_rows.append({
            "detector": detector_name,
            **metrics,
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "f1",
            ascending=False,
        )
    )

    # ================================================
    # PER DATASET METRICS
    # ================================================

    per_dataset_rows = []

    grouped = predictions_df.groupby(
        ["dataset", "detector"],
        sort=False,
    )

    for (
        dataset_name,
        detector_name,
    ), group in grouped:

        metrics = calculate_metrics(
            group
        )

        per_dataset_rows.append({
            "dataset": dataset_name,
            "detector": detector_name,
            **metrics,
        })

    per_dataset_df = pd.DataFrame(
        per_dataset_rows
    )

    # ================================================
    # ERRORS
    # ================================================

    errors_df = predictions_df[
        predictions_df["correct"] == 0
    ].copy()

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    # ================================================
    # SAVE REPORTS
    # ================================================

    predictions_df.to_csv(
        output_dir / "benchmark_predictions.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir / "benchmark_summary.csv",
        index=False,
    )

    per_dataset_df.to_csv(
        output_dir / "benchmark_per_dataset.csv",
        index=False,
    )

    errors_df.to_csv(
        output_dir / "benchmark_errors.csv",
        index=False,
    )

    diagnostics_df.to_csv(
        output_dir / "v2_column_diagnostics.csv",
        index=False,
    )

    # ================================================
    # CONSOLE OUTPUT
    # ================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary_df.round(4).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ERRORS")
    print("=" * 70)

    print(
        errors_df[
            [
                "dataset",
                "column",
                "detector",
                "error_type",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Saved to: {output_dir}")

    print(
        "Evaluated datasets:",
        predictions_df["dataset"].nunique(),
    )

    if args.max_rows is not None:
        print(
            f"WARNING: only first "
            f"{args.max_rows} rows were used"
        )


if __name__ == "__main__":
    main()
