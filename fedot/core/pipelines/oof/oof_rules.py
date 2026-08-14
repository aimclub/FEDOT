from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import StratifiedKFold

from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import _are_stratification_allowed
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass(frozen=True)
class OOFSplit:
    fold_id: int
    train_ids: np.ndarray
    test_ids: np.ndarray


def build_oof_splits(data: InputData,
                     cv_folds: int,
                     shuffle: bool = True,
                     stratify: bool = True,
                     random_seed: int = 42) -> List[OOFSplit]:
    if data.data_type is not DataTypesEnum.table:
        raise ValueError('OOF pipeline evaluation is supported only for tabular InputData.')
    if data.task.task_type not in (TaskTypesEnum.classification, TaskTypesEnum.regression):
        raise ValueError('OOF pipeline evaluation is supported only for classification and regression tasks.')
    if cv_folds is None or cv_folds < 2:
        raise ValueError('OOF pipeline evaluation requires cv_folds >= 2.')
    if cv_folds > data.target.shape[0] - 1:
        raise ValueError(f'cv_folds ({cv_folds}) is greater than the maximum allowed count {data.target.shape[0] - 1}')

    stratify = stratify and _are_stratification_allowed(data, split_ratio=1 - 1 / (cv_folds + 1))
    if data.task.task_type is TaskTypesEnum.classification and stratify:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_seed if shuffle else None)

    return [
        OOFSplit(fold_id=fold_id, train_ids=train_ids, test_ids=test_ids)
        for fold_id, (train_ids, test_ids) in enumerate(splitter.split(data.target, data.target))
    ]
