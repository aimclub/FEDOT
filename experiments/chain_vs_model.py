from typing import Tuple

import numpy as np
from numpy import ndarray

from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import TaskTypesEnum, Task
from experiments.synth_generator.generators.mdc import generated_dataset


def default_mdc_dataset():
    params = {
        'n_samples': 2000,
        'n_feats': 2,
        'k': 3,
        'min_samples': 0,
        'possible_distributions': ['gaussian', 'gamma'],
        'corr': 0.,
        'compactness_factor': 0.1,
        'alpha_n': 1,
        'outliers': 50,
        'ki_coeff3': 3.
    }

    samples, labels = generated_dataset(params)

    return samples, labels


def fedot_input_data_format(mdc_dataset: Tuple[ndarray, ndarray]):
    samples, labels = mdc_dataset

    idx = np.arange(0, samples.shape[0])
    target = labels.reshape(-1)

    fedot_input = InputData(idx=idx, features=samples, target=target,
                            task=Task(TaskTypesEnum.classification),
                            data_type=DataTypesEnum.table)
    return fedot_input


fedot_input_data_format(default_mdc_dataset())
