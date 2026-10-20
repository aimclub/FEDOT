"""Prepare and apply manual label revisions without changing the source CSV."""

import argparse
from pathlib import Path

import pandas as pd

LABELS_DIR = Path(__file__).resolve().parent / 'labels'
SOURCE = LABELS_DIR / 'benchmark_columns_to_label.csv'
REVIEW = LABELS_DIR / 'label_review.csv'
REVISED = LABELS_DIR / 'benchmark_columns_to_label_revised.csv'
CHANGES = LABELS_DIR / 'label_changes.csv'
KEYS = ['dataset', 'column']


def load_labels() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    required = set(KEYS + ['target'])
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate dataset/column keys')
    if not frame['target'].isin(['0', '1']).all():
        raise ValueError('Targets must be 0 or 1')
    return frame


def prepare(force: bool = False) -> None:
    if REVIEW.exists() and not force:
        raise FileExistsError(f'{REVIEW} already exists; pass --force to overwrite it')
    frame = load_labels().copy()
    frame.insert(3, 'reviewed_target', '')
    frame.to_csv(REVIEW, index=False, encoding='utf-8-sig')
    print(f'Edit reviewed_target in: {REVIEW}')


def apply() -> None:
    original = load_labels()
    reviewed = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    if reviewed.duplicated(KEYS).any():
        raise ValueError('Duplicate keys in review')
    if not reviewed['reviewed_target'].isin(['', '0', '1']).all():
        raise ValueError('reviewed_target must be blank, 0, or 1')

    if not original[KEYS].sort_values(KEYS).reset_index(drop=True).equals(
        reviewed[KEYS].sort_values(KEYS).reset_index(drop=True)
    ):
        raise ValueError('The set of dataset/column keys changed')

    merged = original.merge(
        reviewed[KEYS + ['reviewed_target']], on=KEYS, validate='one_to_one'
    )
    changed = merged['reviewed_target'].isin(['0', '1'])
    revised = original.copy()
    revised['target'] = merged['target'].where(~changed, merged['reviewed_target'])

    changes = merged.loc[changed & (merged['target'] != revised['target']), KEYS + ['target']]
    changes = changes.copy()
    changes['new_target'] = revised.loc[changes.index, 'target']

    revised.to_csv(REVISED, index=False, encoding='utf-8-sig')
    changes.to_csv(CHANGES, index=False, encoding='utf-8-sig')
    print(f'Revised labels: {REVISED}; changed columns: {len(changes)}')
    print('The original CSV was not overwritten.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'apply'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(force=args.force)
    else:
        apply()


if __name__ == '__main__':
    main()
