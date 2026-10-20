"""Tests for manual benchmark labels and review files."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiment.text_detection import labels
from experiment.text_detection.benchmark import expected_labels, load_manual_labels


def write_csv(path: Path, records: list) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def test_column_names_keep_leading_spaces(tmp_path):
    path = tmp_path / 'labels.csv'
    write_csv(path, [
        {'dataset': 'news_channel', 'column': ' num_keywords', 'target': 0},
    ])
    result = load_manual_labels(path)
    assert result == {('news_channel', ' num_keywords'): 0}
    assert ('news_channel', 'num_keywords') not in result


def test_label_validation(tmp_path):
    path = tmp_path / 'labels.csv'
    write_csv(path, [
        {'dataset': 'sample', 'column': 'description', 'target': 1},
        {'dataset': 'sample', 'column': 'description', 'target': 0},
    ])
    with pytest.raises(ValueError, match='Duplicate'):
        load_manual_labels(path)

    write_csv(path, [
        {'dataset': 'sample', 'column': 'description', 'target': ''},
    ])
    with pytest.raises(ValueError, match='0 or 1'):
        load_manual_labels(path)

    write_csv(path, [{'dataset': 'sample', 'column': 'description'}])
    with pytest.raises(ValueError, match='Missing label columns'):
        load_manual_labels(path)


def test_benchmark_provided_types_take_precedence():
    dataset = SimpleNamespace(
        feature_columns=['headline', 'category'],
        feature_types=['text', 'categorical'],
    )
    manual = {('sample', 'headline'): 0, ('sample', 'category'): 1}
    assert expected_labels('sample', dataset, manual) == (
        {'headline': 1, 'category': 0}, []
    )


def test_missing_feature_types_use_manual_labels():
    dataset = SimpleNamespace(feature_columns=['title', 'category'])
    manual = {('sample', 'title'): 1, ('sample', 'category'): 0}
    assert expected_labels('sample', dataset, manual) == (
        {'title': 1, 'category': 0}, []
    )


def test_incomplete_manual_labels_are_reported():
    dataset = SimpleNamespace(feature_columns=['title', 'category'])
    manual = {('sample', 'title'): 1}
    assert expected_labels('sample', dataset, manual) == (
        {'title': 1}, ['category']
    )


def test_invalid_feature_types_fall_back_to_manual():
    dataset = SimpleNamespace(feature_columns=['title', 'category'], feature_types=['text'])
    manual = {('sample', 'title'): 0, ('sample', 'category'): 0}
    assert expected_labels('sample', dataset, manual) == (
        {'title': 0, 'category': 0}, []
    )


def test_review_round_trip_does_not_overwrite_source(tmp_path, monkeypatch):
    source = tmp_path / 'source.csv'
    review = tmp_path / 'review.csv'
    revised = tmp_path / 'revised.csv'
    changes = tmp_path / 'changes.csv'
    write_csv(source, [
        {'dataset': 'first', 'column': 'title', 'target': 1},
        {'dataset': 'first', 'column': 'category', 'target': 0},
    ])
    monkeypatch.setattr(labels, 'SOURCE', source)
    monkeypatch.setattr(labels, 'REVIEW', review)
    monkeypatch.setattr(labels, 'REVISED', revised)
    monkeypatch.setattr(labels, 'CHANGES', changes)

    labels.prepare()
    prepared = pd.read_csv(review, dtype=str, keep_default_na=False)
    prepared.loc[prepared['column'] == 'title', 'reviewed_target'] = '0'
    prepared.to_csv(review, index=False)
    labels.apply()

    assert pd.read_csv(source)['target'].tolist() == [1, 0]
    assert pd.read_csv(revised)['target'].tolist() == [0, 0]
    changed = pd.read_csv(changes)
    assert changed[['column', 'target', 'new_target']].iloc[0].tolist() == ['title', 1, 0]
    with pytest.raises(FileExistsError):
        labels.prepare()


def test_review_rejects_unknown_target(tmp_path, monkeypatch):
    source = tmp_path / 'source.csv'
    review = tmp_path / 'review.csv'
    write_csv(source, [{'dataset': 'a', 'column': 'title', 'target': 1}])
    write_csv(review, [
        {'dataset': 'a', 'column': 'title', 'reviewed_target': 'unknown'},
    ])
    monkeypatch.setattr(labels, 'SOURCE', source)
    monkeypatch.setattr(labels, 'REVIEW', review)
    with pytest.raises(ValueError, match='reviewed_target'):
        labels.apply()


def test_review_rejects_changed_columns(tmp_path, monkeypatch):
    source = tmp_path / 'source.csv'
    review = tmp_path / 'review.csv'
    write_csv(source, [{'dataset': 'a', 'column': 'title', 'target': 1}])
    write_csv(review, [
        {'dataset': 'a', 'column': 'different', 'reviewed_target': '0'},
    ])
    monkeypatch.setattr(labels, 'SOURCE', source)
    monkeypatch.setattr(labels, 'REVIEW', review)
    with pytest.raises(ValueError, match='keys changed'):
        labels.apply()


def test_committed_manual_labels_are_valid():
    from experiment.text_detection.benchmark import LABELS_PATH

    manual = load_manual_labels(LABELS_PATH)
    assert manual
    assert set(manual.values()) <= {0, 1}
    assert ('news_channel', ' num_keywords') in manual
