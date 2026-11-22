"""Contracts for prospective probes; no remote data or expensive model fits."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

from examples.advanced.diagnostics.weak_dataset_probe import (
    split_rows, validate_probabilities, write_json, run_cell,
)


def test_partitions_cover_rows_once_preserve_classes_and_are_reproducible():
    y = np.repeat(np.arange(7), 100)
    a, b = split_rows(y), split_rows(y)
    assert set(a) == {'fit', 'select', 'confirm'}
    assert np.array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(len(y)))
    for name in a:
        assert np.array_equal(a[name], b[name])
        assert np.array_equal(np.unique(y[a[name]]), np.arange(7))
    assert [len(a[k]) for k in ('fit', 'select', 'confirm')] == [420, 140, 140]


@pytest.mark.parametrize('p,classes', [
    ([[np.nan, .5]], [0, 1]), ([[1.1, -.1]], [0, 1]),
    ([[.2, .3]], [0, 1]), ([[.1, .2, .7]], [0, 1]),
    ([[.5, .5]], [1, 0]), ([[.5, .5]], [0, 0]),
])
def test_bad_probabilities_or_class_order_fail_closed(p, classes):
    with pytest.raises(ValueError):
        validate_probabilities(p, [0], classes)


def test_loss_matches_manual_likelihood_and_rejects_unknown_truth():
    p = np.array([[.9, .1], [.25, .75]])
    assert validate_probabilities(p, [0, 1], [0, 1]) == pytest.approx(-np.log([.9, .75]).mean())
    with pytest.raises(ValueError):
        validate_probabilities(p, [0, 2], [0, 1])


def test_existing_artifacts_are_never_overwritten(tmp_path):
    path = tmp_path / 'result.json'
    write_json(path, {'old': True})
    with pytest.raises(FileExistsError):
        write_json(path, {'new': True})
    assert json.loads(path.read_text()) == {'old': True}


def test_exhausted_budget_does_not_spawn_or_create_a_spec(tmp_path):
    result = run_cell(Path('missing-data.npz'), tmp_path, 'never', {}, 10, time.monotonic() - 1)
    assert result['status'] == 'not_started_budget'
    assert list(tmp_path.iterdir()) == []


def test_json_rejects_nonfinite_metrics(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / 'bad.json', {'loss': float('inf')})
