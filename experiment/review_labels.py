
import argparse
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent

LABELS = EXPERIMENT_DIR / "benchmark_columns_to_label.csv"
REVIEW = EXPERIMENT_DIR / "benchmark_label_review.csv"
REVIEWED = EXPERIMENT_DIR / "benchmark_columns_to_label_reviewed.csv"
CHANGES = EXPERIMENT_DIR / "benchmark_label_changes.csv"

KEYS = ["dataset", "column"]

AMBIGUOUS = (
    "title", "name", "summary", "keyword", "skill",
    "attribute", "actor", "director", "author",
    "amenit", "headline", "tag", "genre",
    "location", "address", "job_desig",
    "product", "category",
)

STRUCTURED = (
    "url", "email", "uuid", "identifier",
    "filepath", "file_path", "_id", "id_",
    "_code", "code_", "phone", "zip", "postal",
)


def load_labels() -> pd.DataFrame:
    """Load and validate existing benchmark labels."""

    if not LABELS.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS}"
        )

    df = pd.read_csv(
        LABELS,
        dtype=str,
        keep_default_na=False,
    )

    required = set(KEYS + ["target"])

    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate dataset/column pairs")

    if not df["target"].isin(["0", "1"]).all():
        raise ValueError("Target must contain only 0 or 1")

    return df


def rank_column(row: pd.Series) -> tuple[str, str]:
    """Assign a review priority to a column."""

    name = str(row["column"]).lower()
    dtype = str(row.get("dtype", "")).lower()
    target = str(row["target"])

    examples = [
        str(row.get(f"example_{i}", "")).strip()
        for i in range(1, 4)
    ]

    sample = next(
        (value for value in examples if value),
        "",
    )

    if target == "1" and any(
        item in dtype for item in ("int", "float", "bool")
    ):
        return (
            "high",
            "Numeric/boolean column is labeled as text",
        )

    if target == "1" and sample.startswith(("{", "[")):
        return (
            "high",
            "Possibly structured list or dictionary",
        )

    if any(item in name for item in AMBIGUOUS):
        return (
            "high",
            "Ambiguous: check whether NLP processing is useful",
        )

    if target == "1" and any(
        item in name for item in STRUCTURED
    ):
        return (
            "high",
            "Structured field may be incorrectly labeled text",
        )

    if "object" in dtype or "string" in dtype:
        return (
            "medium",
            "String column: verify text vs categorical",
        )

    return (
        "low",
        "Numeric or other column",
    )


def prepare(force: bool = False):
    """Generate a CSV for manual label review."""

    if REVIEW.exists() and not force:
        raise FileExistsError(
            f"{REVIEW} already exists. "
            "Use --force to recreate it."
        )

    df = load_labels().copy()

    priorities = df.apply(
        rank_column,
        axis=1,
    )

    df.insert(
        3,
        "current_target",
        df.pop("target"),
    )

    df.insert(
        4,
        "reviewed_target",
        "",
    )

    df.insert(
        5,
        "priority",
        [item[0] for item in priorities],
    )

    df.insert(
        6,
        "reason",
        [item[1] for item in priorities],
    )

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    df["_order"] = df["priority"].map(priority_order)

    df = (
        df.sort_values(
            ["_order", "dataset", "column"]
        )
        .drop(columns="_order")
    )

    df.to_csv(
        REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nREVIEW FILE CREATED")
    print(REVIEW)

    print("\nPriority distribution:")
    print(df["priority"].value_counts())

    print("\nFill reviewed_target:")
    print("1 = semantic text")
    print("0 = not text")
    print("empty = not reviewed")


def apply():
    """Apply reviewed labels without overwriting the original."""

    original = load_labels()

    if not REVIEW.exists():
        raise FileNotFoundError(
            f"Review file not found: {REVIEW}"
        )

    reviews = pd.read_csv(
        REVIEW,
        dtype=str,
        keep_default_na=False,
    )

    required = set(
        KEYS + ["current_target", "reviewed_target"]
    )

    if not required.issubset(reviews.columns):
        raise ValueError(
            f"Missing review columns: {required - set(reviews.columns)}"
        )

    if reviews.duplicated(KEYS).any():
        raise ValueError("Duplicate dataset/column pairs")

    if not reviews["reviewed_target"].isin(
        ["", "0", "1"]
    ).all():
        raise ValueError(
            "reviewed_target must be empty, 0 or 1"
        )

    original_keys = set(
        map(
            tuple,
            original[KEYS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    review_keys = set(
        map(
            tuple,
            reviews[KEYS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    if original_keys != review_keys:
        raise ValueError(
            "Dataset/column pairs have changed"
        )

    merged = original.merge(
        reviews[
            KEYS + ["current_target", "reviewed_target"]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if not (
        merged["target"] == merged["current_target"]
    ).all():
        raise ValueError(
            "Original labels changed after preparing review"
        )

    checked = merged["reviewed_target"].isin(["0", "1"])

    updated = merged["target"].where(
        ~checked,
        merged["reviewed_target"],
    )

    changed = checked & (merged["target"] != updated)

    changes = merged.loc[
        changed,
        KEYS + ["target"],
    ].copy()

    changes["new_target"] = updated.loc[changes.index]

    result = original.copy()
    result["target"] = updated

    result.to_csv(
        REVIEWED,
        index=False,
        encoding="utf-8-sig",
    )

    changes.to_csv(
        CHANGES,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nREVIEW COMPLETED")
    print(f"Reviewed: {checked.sum()} / {len(original)}")
    print(f"Changed labels: {changed.sum()}")

    print(f"\nUpdated labels: {REVIEWED}")
    print(f"Changes: {CHANGES}")

    print(
        "\nUnreviewed columns retain their previous labels."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["prepare", "apply"],
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.force)

    elif args.command == "apply":
        apply()
