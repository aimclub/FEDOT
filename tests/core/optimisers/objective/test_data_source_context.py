import numpy as np

from fedot.core.data.input_data.data import InputData
from fedot.core.optimisers.objective.data_source_context import (
    ComposerDataSourceMode,
    build_external_holdout_composer_data_source_context,
    build_internal_composer_data_source_context,
)
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _classification_data(size=10):
    return InputData(
        idx=np.arange(size),
        features=np.arange(size * 2).reshape(size, 2),
        target=np.array([0, 1] * (size // 2)),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_internal_composer_data_source_context_uses_cv_folds():
    data = _classification_data()

    context = build_internal_composer_data_source_context(data, cv_folds=2)
    folds = list(context.data_producer())

    assert context.mode is ComposerDataSourceMode.internal_split
    assert context.validation_blocks is None
    assert len(folds) == 2


def test_external_holdout_composer_data_source_context_uses_common_validation_data():
    train_data = _classification_data(size=8)
    validation_data = _classification_data(size=4)

    context = build_external_holdout_composer_data_source_context(train_data, validation_data)
    folds = list(context.data_producer())

    assert context.mode is ComposerDataSourceMode.external_holdout
    assert context.validation_blocks is None
    assert folds == [(train_data, validation_data)]
