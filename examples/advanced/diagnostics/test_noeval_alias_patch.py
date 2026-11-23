from pathlib import Path

import pytest

from examples.advanced.diagnostics.noeval_alias_patch import patched_source


def test_patch_is_scoped_and_preserves_other_boosters():
    path=Path(__file__).resolve().parents[3]/'fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py'
    source=path.read_text()
    result=patched_source(source)
    assert result.split('class FedotCatBoostImplementation')[0]==source.split('class FedotCatBoostImplementation')[0]
    assert result.split('class FedotCatBoostClassificationImplementation')[1]==source.split('class FedotCatBoostClassificationImplementation')[1]
    compile(result,str(path),'exec')
    assert "self.params.update(use_best_model=False, early_stopping_rounds=False)" not in result


def test_patch_refuses_reapplication_or_unknown_source():
    with pytest.raises((AssertionError,ValueError)):
        patched_source('unrelated source')
