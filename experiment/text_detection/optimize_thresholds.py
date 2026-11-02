
"""Optimize FEDOT TextDataDetector thresholds on the multimodal benchmark.

Run from the FEDOT repository root, after placing this file at
experiment/text_detection/optimize_thresholds.py:

    python -m experiment.text_detection.optimize_thresholds \
        --trials 150 --folds 5 --max-rows 1000 --verify

Reports are written to:
    experiment/text_detection/reports/threshold_optimization/

The grouped out-of-fold estimate uses whole datasets as held-out groups.
The final thresholds are tuned on all supplied development datasets.
This benchmark was previously used to develop V2; results are not a fully
independent measurement of generalization to entirely new datasets.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from fedot.core.data.data_detection import TextDataDetector
from experiment.text_detection.benchmark import (
    LABELS_PATH,
    REPORTS_DIR,
    expected_labels,
    load_benchmark_registry,
    load_manual_labels,
)


STRUCTURED_FEATURES = (
    'url_ratio', 'email_ratio', 'uuid_ratio', 'numeric_ratio',
    'path_ratio', 'id_ratio',
)

COLLECTION_FEATURES = (
    'collection_ratio', 'slug_ratio', 'person_list_ratio',
)

NUMERIC_FEATURES = (
    *STRUCTURED_FEATURES,
    *COLLECTION_FEATURES,
    'median_word_count', 'median_length', 'mean_word_count',
    'mean_length', 'alphabetic_ratio', 'unique_ratio',
    'vocabulary_size', 'function_word_ratio',
)

# Match the default V2 detector exactly before optimizing any thresholds.
DEFAULT_PARAMS = {
    'structured_threshold': 0.80,
    'collection_threshold': 0.60,
    'short_unique_ratio': 0.45,
    'short_min_vocabulary': 15,
    'short_min_words': 2.4,
}

# Candidate search domain; not a claim that these limits are optimal.
BOUNDS = {
    'structured_threshold': (0.60, 0.95),
    'collection_threshold': (0.40, 0.90),
    'short_unique_ratio': (0.20, 0.80),
    'short_min_vocabulary': (5, 50),
    'short_min_words': (1.6, 4.0),
}


def column_features(
    detector: TextDataDetector,
    column: pd.Series,
) -> Dict[str, float]:
    """Compute expensive column statistics only once."""
    result = {'eligible': False}

    if not (
        pd.api.types.is_object_dtype(column)
        or pd.api.types.is_string_dtype(column)
    ):
        return result

    values = column.dropna().astype(str).str.strip()
    values = values[values != '']

    if len(values) < detector.MIN_VALUES:
        return result

    if detector._is_float_compatible(column):
        return result

    result.update(detector.extract_features(values))

    result.update({
        'collection_ratio': float(values.map(
            detector._is_serialized_collection
        ).mean()),
        'slug_ratio': float(values.map(detector._is_slug).mean()),
        'person_list_ratio': float(values.map(
            detector._is_person_list
        ).mean()),
    })

    result['eligible'] = True
    return result


def load_features(
    split: str,
    max_rows: Optional[int],
    seed: int,
    verify: bool,
) -> pd.DataFrame:
    """Load the real benchmark and compute column features and labels."""
    registry, aliases = load_benchmark_registry()
    manual = load_manual_labels(LABELS_PATH)
    detector = TextDataDetector()
    records: List[dict] = []

    for dataset_name in dict.fromkeys(aliases.values()):
        print(f'Extracting features: {dataset_name}', flush=True)

        dataset = registry.create(dataset_name, split)
        columns = list(dataset.feature_columns)

        labels, missing = expected_labels(dataset_name, dataset, manual)

        if missing:
            raise ValueError(
                f'{dataset_name}: missing manual labels: {missing}'
            )

        frame = dataset.data[columns]

        if max_rows is not None and len(frame) > max_rows:
            # Fixed seed and random rows, not just the first N rows.
            frame = frame.sample(n=max_rows, random_state=seed)

        for name in columns:
            series = frame[name]

            row = {
                'dataset': dataset_name,
                'column': name,
                'expected': int(labels[name]),
                'sample_rows': len(series),
            }

            row.update(column_features(detector, series))

            if verify:
                row['production_prediction'] = int(
                    detector.analyze_column(series)['is_text']
                )

            records.append(row)

    result = pd.DataFrame.from_records(records)

    if result.empty:
        raise ValueError('The benchmark returned no columns')

    if result['dataset'].nunique() < 2:
        raise ValueError('At least two datasets are required')

    result['eligible'] = result['eligible'].fillna(False).astype(bool)

    for name in NUMERIC_FEATURES:
        result[name] = pd.to_numeric(
            result.get(name, 0.0), errors='coerce'
        ).fillna(0.0)

    if result.duplicated(['dataset', 'column']).any():
        raise ValueError('Duplicate dataset/column pairs')

    return result


def predict(frame: pd.DataFrame, params: dict) -> np.ndarray:
    """Vectorized version of the production V2 decision rules.

    All long-text and repeated-text thresholds remain fixed. Only the
    five thresholds listed in DEFAULT_PARAMS are optimized.
    """
    eligible = frame['eligible'].to_numpy(dtype=bool)

    structured = frame[list(STRUCTURED_FEATURES)].ge(
        params['structured_threshold']
    ).any(axis=1).to_numpy()

    collection = frame[list(COLLECTION_FEATURES)].ge(
        params['collection_threshold']
    ).any(axis=1).to_numpy()

    def v(name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=float)

    words = v('mean_word_count')
    length = v('mean_length')
    alpha = v('alphabetic_ratio')
    unique = v('unique_ratio')
    vocabulary = v('vocabulary_size')

    long_text = (
        (
            (v('median_word_count') >= 7)
            & (v('median_length') >= 40)
            & (alpha >= 0.50)
        )
        | ((words >= 5) & (length >= 50) & (alpha >= 0.52))
    )

    short_text = (
        (unique >= params['short_unique_ratio'])
        & (words >= params['short_min_words'])
        & (length >= 12)
        & (alpha >= 0.48)
        & (vocabulary >= params['short_min_vocabulary'])
    )

    repeated_text = (
        (words >= 3.4)
        & (length >= 18)
        & (alpha >= 0.65)
        & (vocabulary >= 40)
        & (
            (unique >= 0.03)
            | (v('function_word_ratio') >= 0.02)
            | (words >= 3.8)
        )
    )

    return (
        eligible
        & ~structured
        & ~collection
        & (long_text | short_text | repeated_text)
    ).astype(int)


def get_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict:
    """Return confusion matrix counts and classification metrics."""
    y = np.asarray(expected, dtype=int)
    p = np.asarray(predicted, dtype=int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0.0
    )

    return {
        'columns': len(y),
        'text_columns': int(y.sum()),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def normalized_distance(params: dict) -> float:
    """Prefer less disruptive changes when multiple trials share best F1."""
    return sum(
        abs(params[name] - default) / (BOUNDS[name][1] - BOUNDS[name][0])
        for name, default in DEFAULT_PARAMS.items()
    )


def tune(
    train: pd.DataFrame,
    trials: int,
    seed: int,
):
    """Tune only on the supplied training datasets."""
    y = train['expected'].to_numpy(dtype=int)

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
    )

    study.enqueue_trial(DEFAULT_PARAMS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'structured_threshold': trial.suggest_float(
                'structured_threshold', 0.60, 0.95, step=0.05
            ),
            'collection_threshold': trial.suggest_float(
                'collection_threshold', 0.40, 0.90, step=0.05
            ),
            'short_unique_ratio': trial.suggest_float(
                'short_unique_ratio', 0.20, 0.80, step=0.05
            ),
            'short_min_vocabulary': trial.suggest_int(
                'short_min_vocabulary', 5, 50, step=5
            ),
            'short_min_words': trial.suggest_float(
                'short_min_words', 1.6, 4.0, step=0.2
            ),
        }

        return get_metrics(
            y,
            predict(train, params)
        )['f1']

    study.optimize(
        objective,
        n_trials=trials,
        show_progress_bar=False,
    )

    best = max(
        trial.value
        for trial in study.trials
        if trial.value is not None
    )

    candidates = [
        trial
        for trial in study.trials
        if trial.value is not None
        and abs(trial.value - best) < 1e-12
    ]

    selected = min(
        candidates,
        key=lambda trial: normalized_distance(trial.params),
    )

    return selected.params, study


def write_threshold_sensitivity(
    frame: pd.DataFrame,
    best_params: dict,
    output_file: Path,
) -> None:
    """One-at-a-time threshold scan, globally and for each dataset.

    This is development-set sensitivity, not independent test performance.
    """
    steps = {
        'structured_threshold': 0.05,
        'collection_threshold': 0.05,
        'short_unique_ratio': 0.05,
        'short_min_vocabulary': 5,
        'short_min_words': 0.2,
    }

    groups = [('ALL', frame)] + list(frame.groupby('dataset'))

    rows = []

    for name, (lower, upper) in BOUNDS.items():
        step = steps[name]
        values = np.arange(lower, upper + step / 2, step)

        for value in values:
            candidate = dict(best_params)

            candidate[name] = (
                int(round(value))
                if name == 'short_min_vocabulary'
                else round(float(value), 3)
            )

            for dataset, group in groups:
                predicted = predict(group, candidate)

                metrics = get_metrics(
                    group['expected'].to_numpy(),
                    predicted,
                )

                rows.append({
                    'dataset': dataset,
                    'parameter': name,
                    'value': candidate[name],
                    **metrics,
                })

    pd.DataFrame(rows).to_csv(output_file, index=False)


def run(
    features: pd.DataFrame,
    trials: int,
    folds: int,
    seed: int,
    output_dir: Path,
    verify: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_groups = features['dataset'].nunique()

    if folds < 2 or folds > n_groups:
        raise ValueError(f'--folds must be in [2, {n_groups}]')

    if trials < 1:
        raise ValueError('--trials must be positive')

    baseline = predict(features, DEFAULT_PARAMS)

    if verify:
        if 'production_prediction' not in features.columns:
            raise ValueError('--verify requires raw benchmark data')

        live = features['production_prediction'].to_numpy(dtype=int)
        mismatches = np.flatnonzero(live != baseline)

        if len(mismatches):
            bad = features.iloc[mismatches][['dataset', 'column']]

            raise AssertionError(
                'Cached rules differ from production detector:\n'
                + bad.to_string(index=False)
                + '\nUpdate predict() to match analyze_column() first.'
            )

        print('Production V2 rules verified against cached predictions.')

    features.to_csv(
        output_dir / 'column_features.csv',
        index=False,
    )

    splitter = GroupKFold(n_splits=folds)

    groups = features['dataset'].to_numpy()
    labels = features['expected'].to_numpy(dtype=int)

    out_of_fold = np.zeros(len(features), dtype=int)
    fold_rows = []
    params_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups),
        start=1,
    ):
        train = features.iloc[train_idx]
        test = features.iloc[test_idx]

        print(
            f'Fold {fold}/{folds}: train={train["dataset"].nunique()} '
            f'datasets, held out={test["dataset"].nunique()} datasets',
            flush=True,
        )

        params, _ = tune(
            train,
            trials=trials,
            seed=seed + fold,
        )

        pred = predict(test, params)
        out_of_fold[test_idx] = pred

        base_metrics = get_metrics(
            labels[test_idx],
            baseline[test_idx],
        )

        tuned_metrics = get_metrics(
            labels[test_idx],
            pred,
        )

        fold_rows.extend([
            {
                'fold': fold,
                'method': 'original',
                **base_metrics,
            },
            {
                'fold': fold,
                'method': 'optimized',
                **tuned_metrics,
            },
        ])

        params_rows.append({
            'fold': fold,
            **params,
        })

    output = features[
        ['dataset', 'column', 'expected']
    ].copy()

    output['baseline'] = baseline
    output['optimized_oof'] = out_of_fold

    output.to_csv(
        output_dir / 'out_of_fold_predictions.csv',
        index=False,
    )

    pd.DataFrame(fold_rows).to_csv(
        output_dir / 'outer_fold_metrics.csv',
        index=False,
    )

    pd.DataFrame(params_rows).to_csv(
        output_dir / 'outer_fold_parameters.csv',
        index=False,
    )

    # Final setting: optimize on all development datasets for deployment.
    final_params, final_study = tune(
        features,
        trials=trials,
        seed=seed,
    )

    final_pred = predict(features, final_params)

    write_threshold_sensitivity(
        features,
        final_params,
        output_dir / 'threshold_sensitivity.csv',
    )

    final_study.trials_dataframe().to_csv(
        output_dir / 'optuna_trials.csv',
        index=False,
    )

    per_dataset = []

    for dataset, group in features.groupby('dataset', sort=False):
        positions = group.index.to_numpy()

        for method, p in (
            ('original', baseline),
            ('optimized_oof', out_of_fold),
        ):
            per_dataset.append({
                'dataset': dataset,
                'method': method,
                **get_metrics(
                    labels[positions],
                    p[positions],
                ),
            })

    pd.DataFrame(per_dataset).to_csv(
        output_dir / 'per_dataset_metrics.csv',
        index=False,
    )

    summary = {
        'note': (
            'Grouped OOF assessment holds out whole datasets. '
            'The final fit uses all development datasets; '
            'its F1 is in-sample. '
            'These datasets were used during prior development of V2.'
        ),
        'datasets': int(n_groups),
        'columns': int(len(features)),
        'baseline_params': DEFAULT_PARAMS,
        'best_params': final_params,
        'baseline': get_metrics(labels, baseline),
        'optimized_grouped_oof': get_metrics(
            labels,
            out_of_fold,
        ),
        'optimized_in_sample': get_metrics(
            labels,
            final_pred,
        ),
    }

    with (
        output_dir / 'summary.json'
    ).open('w', encoding='utf-8') as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (
        output_dir / 'best_params.json'
    ).open('w', encoding='utf-8') as file:
        json.dump(
            final_params,
            file,
            indent=2,
        )

    print('\nFinal thresholds:')
    print(json.dumps(final_params, indent=2))

    print('\nGrouped out-of-fold metrics (held-out datasets):')

    print(
        pd.DataFrame([
            {
                'method': 'original',
                **summary['baseline'],
            },
            {
                'method': 'optimized',
                **summary['optimized_grouped_oof'],
            },
        ]).round(4).to_string(index=False)
    )

    print(
        '\nFinal optimization F1 on development data: '
        f"{summary['optimized_in_sample']['f1']:.4f} (not hold-out)"
    )

    print(f'Reports: {output_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--split',
        choices=('train', 'test'),
        default='train',
    )

    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=150,
    )

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--verify',
        action='store_true',
    )

    parser.add_argument(
        '--features-csv',
        type=Path,
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=REPORTS_DIR / 'threshold_optimization',
    )

    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 3:
        parser.error('--max-rows must be >= 3')

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.features_csv:
        frame = pd.read_csv(
            args.features_csv,
            keep_default_na=False,
        )

        required = {
            'dataset',
            'column',
            'expected',
            'eligible',
            *NUMERIC_FEATURES,
        }

        if missing := required - set(frame.columns):
            raise ValueError(
                f'Missing feature columns: {sorted(missing)}'
            )

        if not frame['expected'].astype(str).isin(('0', '1')).all():
            raise ValueError('Expected labels must be 0/1')

        frame['expected'] = frame['expected'].astype(int)

        frame['eligible'] = (
            frame['eligible'].astype(str).str.lower().eq('true')
        )

        for name in NUMERIC_FEATURES:
            frame[name] = pd.to_numeric(
                frame[name],
                errors='raise',
            )

        frame = frame.reset_index(drop=True)

    else:
        frame = load_features(
            split=args.split,
            max_rows=args.max_rows,
            seed=args.seed,
            verify=args.verify,
        )

    run(
        features=frame,
        trials=args.trials,
        folds=args.folds,
        seed=args.seed,
        output_dir=args.output,
        verify=args.verify,
    )


if __name__ == '__main__':
    main()
