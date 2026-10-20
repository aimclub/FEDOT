"""Compare text-column detectors on the multimodal benchmark."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from fedot.core.data.data_detection import TextDataDetector

from experiment.text_detection.detectors import (
    CurrentFedotDetector,
    Pr903Detector,
    TextDataDetectorV1,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
FEDOT_ROOT = EXPERIMENT_DIR.parents[1]
LABELS_PATH = EXPERIMENT_DIR / 'labels' / 'benchmark_columns_to_label.csv'
REPORTS_DIR = EXPERIMENT_DIR / 'reports'
BENCHMARK_SRC = (
    FEDOT_ROOT.parent / 'automl_multimodal_benchmark'
    / 'multimodal_text_benchmark' / 'src'
)


def load_benchmark_registry() -> Tuple[Any, Dict[str, str]]:
    """Load the external benchmark installed next to FEDOT."""
    if not BENCHMARK_SRC.is_dir():
        raise FileNotFoundError(
            f'Benchmark source not found: {BENCHMARK_SRC}. '
            'Clone sxjscience/automl_multimodal_benchmark next to FEDOT.'
        )

    sys.path.insert(0, str(BENCHMARK_SRC))
    from auto_mm_bench.datasets import (
        TEXT_BENCHMARK_ALIAS_MAPPING,
        dataset_registry,
    )

    return dataset_registry, TEXT_BENCHMARK_ALIAS_MAPPING


def load_manual_labels(path: Path) -> Dict[Tuple[str, str], int]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'dataset', 'column', 'target'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing label columns: {sorted(required - set(frame.columns))}')
    if frame.duplicated(['dataset', 'column']).any():
        raise ValueError('Duplicate dataset/column pairs in manual labels')
    if not frame['target'].isin(['0', '1']).all():
        raise ValueError('Manual labels must all be 0 or 1')

    # Some original dataset columns start with a space.
    return {
        (row.dataset, row.column): int(row.target)
        for row in frame.itertuples(index=False)
    }


def expected_labels(
    dataset_name: str,
    dataset: Any,
    manual: Dict[Tuple[str, str], int],
) -> Tuple[Dict[str, int], List[str]]:
    columns = list(dataset.feature_columns)
    types: Optional[List[Any]] = None
    try:
        types = list(dataset.feature_types)
    except (AttributeError, NotImplementedError, TypeError):
        pass

    if types is not None and len(types) == len(columns):
        return {
            column: int(feature_type == 'text')
            for column, feature_type in zip(columns, types)
        }, []

    labels: Dict[str, int] = {}
    missing: List[str] = []
    for column in columns:
        key = (dataset_name, column)
        if key in manual:
            labels[column] = manual[key]
        else:
            missing.append(column)
    return labels, missing


def metrics(group: pd.DataFrame) -> Dict[str, float]:
    expected = group['expected'].astype(int)
    predicted = group['predicted'].astype(int)
    return {
        'columns': len(group),
        'text_columns': int(expected.sum()),
        'tp': int(((expected == 1) & (predicted == 1)).sum()),
        'fp': int(((expected == 0) & (predicted == 1)).sum()),
        'fn': int(((expected == 1) & (predicted == 0)).sum()),
        'accuracy': float(accuracy_score(expected, predicted)),
        'precision': float(precision_score(expected, predicted, zero_division=0)),
        'recall': float(recall_score(expected, predicted, zero_division=0)),
        'f1': float(f1_score(expected, predicted, zero_division=0)),
    }


def evaluate(
    split: str = 'train',
    max_rows: Optional[int] = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if max_rows is not None and max_rows < 3:
        raise ValueError('max_rows must be at least 3')

    registry, aliases = load_benchmark_registry()
    manual = load_manual_labels(LABELS_PATH)
    output_dir = REPORTS_DIR if split == 'train' else REPORTS_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # A failed rerun must not leave an old summary looking like a new result.
    report_names = (
        'benchmark_predictions.csv', 'benchmark_errors.csv',
        'benchmark_per_dataset.csv', 'benchmark_summary.csv',
        'v2_column_diagnostics.csv',
    )
    for report_name in report_names:
        (output_dir / report_name).unlink(missing_ok=True)

    detectors = {
        'current_fedot': CurrentFedotDetector(),
        'pr903': Pr903Detector(),
        'improved_v1': TextDataDetectorV1(),
        'improved_v2': TextDataDetector(),
    }

    predictions: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, str]] = []
    failed_rows: List[Dict[str, str]] = []

    for dataset_name in dict.fromkeys(aliases.values()):
        print(f'Processing: {dataset_name}')
        try:
            dataset = registry.create(dataset_name, split)
            columns = list(dataset.feature_columns)
            labels, missing = expected_labels(dataset_name, dataset, manual)

            if missing:
                print(f'  Missing labels: {len(missing)}')
                missing_rows.extend(
                    {'dataset': dataset_name, 'column': column} for column in missing
                )
                continue

            dataframe = dataset.data[columns]
            if max_rows is not None:
                dataframe = dataframe.head(max_rows)

            dataset_predictions: List[Dict[str, Any]] = []
            dataset_diagnostics: List[Dict[str, Any]] = []
            for detector_name, detector in detectors.items():
                if detector_name == 'improved_v2':
                    selected = set()
                    for column in columns:
                        result = detector.analyze_column(dataframe[column])
                        if result['is_text']:
                            selected.add(column)
                        dataset_diagnostics.append({
                            'dataset': dataset_name,
                            'column': column,
                            'expected': labels[column],
                            'predicted': int(result['is_text']),
                            'reason': result['reason'],
                            **{
                                f'feature_{name}': value
                                for name, value in result['features'].items()
                            },
                        })
                else:
                    selected = set(detector.define_text_columns(dataframe))

                for column in columns:
                    expected = labels[column]
                    predicted = int(column in selected)
                    if expected == predicted:
                        error_type = 'TP' if expected else 'TN'
                    else:
                        error_type = 'FN' if expected else 'FP'
                    dataset_predictions.append({
                        'dataset': dataset_name,
                        'column': column,
                        'detector': detector_name,
                        'expected': expected,
                        'predicted': predicted,
                        'correct': int(expected == predicted),
                        'error_type': error_type,
                    })

            predictions.extend(dataset_predictions)
            diagnostics.extend(dataset_diagnostics)
        except Exception as error:
            failed_rows.append({'dataset': dataset_name, 'error': repr(error)})
            print(f'  Failed: {error}')

    pd.DataFrame(missing_rows, columns=['dataset', 'column']).to_csv(
        output_dir / 'missing_labels.csv', index=False
    )
    pd.DataFrame(failed_rows, columns=['dataset', 'error']).to_csv(
        output_dir / 'failed_datasets.csv', index=False
    )

    if (missing_rows or failed_rows) and not allow_partial:
        raise RuntimeError(
            'Incomplete benchmark: see missing_labels.csv and failed_datasets.csv. '
            'Pass --allow-partial to inspect incomplete results.'
        )
    if not predictions:
        raise RuntimeError('No datasets evaluated')

    frame = pd.DataFrame(predictions)
    frame.to_csv(output_dir / 'benchmark_predictions.csv', index=False)
    frame[frame['correct'] == 0].to_csv(output_dir / 'benchmark_errors.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / 'v2_column_diagnostics.csv', index=False)

    summary = pd.DataFrame([
        {'detector': name, **metrics(group)}
        for name, group in frame.groupby('detector', sort=False)
    ]).sort_values('f1', ascending=False)
    summary.to_csv(output_dir / 'benchmark_summary.csv', index=False)

    per_dataset = pd.DataFrame([
        {'dataset': dataset_name, 'detector': detector_name, **metrics(group)}
        for (dataset_name, detector_name), group
        in frame.groupby(['dataset', 'detector'], sort=False)
    ])
    per_dataset.to_csv(output_dir / 'benchmark_per_dataset.csv', index=False)

    print('\n' + summary.round(4).to_string(index=False))
    print(
        f'\nDatasets: {frame["dataset"].nunique()}, '
        f'columns per detector: {len(frame) // len(detectors)}'
    )
    if missing_rows or failed_rows:
        print('Partial run: some datasets were skipped or failed')
    print(f'Reports: {output_dir}')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    evaluate(split=args.split, max_rows=args.max_rows, allow_partial=args.allow_partial)


if __name__ == '__main__':
    main()
