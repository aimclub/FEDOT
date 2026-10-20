"""Inspect errors and detection reasons from a completed benchmark run."""

import argparse
from pathlib import Path

import pandas as pd

from experiment.text_detection.benchmark import REPORTS_DIR


def display_errors(reports_dir: Path, detector: str) -> None:
    predictions = pd.read_csv(reports_dir / 'benchmark_predictions.csv')
    diagnostics = pd.read_csv(reports_dir / 'v2_column_diagnostics.csv')

    mistakes = predictions.loc[
        (predictions['detector'] == detector)
        & (predictions['expected'] != predictions['predicted'])
    ].copy()
    if mistakes.empty:
        (reports_dir / f'{detector}_errors.csv').unlink(missing_ok=True)
        print(f'No errors for {detector}')
        return

    if detector == 'improved_v2':
        diagnostic_columns = [
            'dataset', 'column', 'reason', 'feature_mean_word_count',
            'feature_mean_length', 'feature_unique_ratio',
            'feature_function_word_ratio', 'feature_pattern_unique_ratio',
        ]
        mistakes = mistakes.merge(
            diagnostics.reindex(columns=diagnostic_columns),
            on=['dataset', 'column'],
            how='left',
            validate='one_to_one',
        )

    output = reports_dir / f'{detector}_errors.csv'
    mistakes.to_csv(output, index=False)
    print(mistakes.to_string(index=False, max_colwidth=75))
    print(f'\nSaved: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--detector', default='improved_v2')
    args = parser.parse_args()
    directory = REPORTS_DIR if args.split == 'train' else REPORTS_DIR / args.split
    display_errors(directory, args.detector)


if __name__ == '__main__':
    main()
