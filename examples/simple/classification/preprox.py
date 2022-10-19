from copy import deepcopy
from pathlib import Path

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.preprocessing.preprocessing import DataPreprocessor

base_path = Path('../../data/openml')
base_path_npy = Path('../../data/openml/npy')


def get_preprocessed_data_folds(nfolds=(0, 1), stage='train',
                                npy_path = base_path_npy,
                                basename = 'credit-g',
                                ):
    def load_npy(ifold: int, stage='train'):
        fname_features = f'{stage}_{basename}_fold{ifold}.npy'
        fname_target = f'{stage}y_{basename}_fold{ifold}.npy'
        features = np.load(str(npy_path / fname_features))
        targets = np.load(str(npy_path / fname_target)).astype(int)
        return features, targets

    all_features = []
    all_targets = []
    for ifold in range(*nfolds):
        print(f'loading fold: {ifold}')
        # train_x, train_y = load_npy(ifold, stage='train')
        features, targets = load_npy(ifold, stage=stage)
        all_features.append(features)
        all_targets.append(targets)

    features = np.concatenate(all_features)
    targets = np.concatenate(all_targets)
    return InputData(task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table,
                     features=features.astype(float),
                     target=_transform_targets(targets),
                     idx=np.arange(len(targets)))


def _transform_targets(targets: np.ndarray):
    targets.astype(int)
    new_targets = np.empty_like(targets).astype(str)
    new_targets[targets == 0] = 'bad'
    new_targets[targets == 1] = 'good'
    return new_targets


def get_preprocessed_data(nfolds=(0, 1), npy_path=base_path_npy):
    train_set = get_preprocessed_data_folds(nfolds=nfolds, stage='train', npy_path=npy_path)
    test_set = get_preprocessed_data_folds(nfolds=nfolds, stage='test', npy_path=npy_path)
    return train_set, test_set


def get_raw_data(path=base_path / 'credit-g.csv', split_ratio=0.9):
    data = InputData.from_csv(path, target_columns='class', index_col=None)
    train_set, test_set = train_test_data_setup(data, split_ratio=split_ratio)
    return train_set, test_set


def preprocess_data(*data_inputs):
    prox = DataPreprocessor()
    # dummy_pipeline = PipelineBuilder().add_sequence('scaling', 'rf').to_pipeline()
    output = []
    for data in data_inputs:
        data_prox = prox.obligatory_prepare_for_fit(data)
        # data_prox = prox.optional_prepare_for_fit(dummy_pipeline, data_prox)
        data_prox = prox._apply_imputation_unidata(data_prox, source_name='default')
        data_prox = prox._apply_categorical_encoding(data_prox, source_name='default')
        output.append(data_prox)
    return tuple(output)


def sort_data_many(*data: InputData):
    return [sort_data_by_column(d) for d in data]


def sort_data_by_column(data: InputData, *, column=4, sort_index=False) -> InputData:
    inds = data.features[:, column].argsort()

    sorted_data = deepcopy(data)
    if sort_index:
        sorted_data.idx = sorted_data.idx[inds]
    sorted_data.features = sorted_data.features[inds]
    sorted_data.target = sorted_data.target[inds]

    return sorted_data
