from copy import copy
from typing import List

import numpy as np


def prepare_lagged_ts_for_prediction(data: 'InputData', is_for_fit: bool = True):
    criteria = ['features']

    # nan in target is not acceptable in fit step
    if is_for_fit:
        criteria.append('target')
        if data.features is not None:
            data.features = data.features[0:len(data.target), ...]

    cleaned_data = _clean_nans_in_lagged_features(data, criteria)

    return cleaned_data


def _clean_nans_in_lagged_features(data: 'InputData', criteria: List[str]):
    """Removes data items that corresponds to NaNs in array_with_nans
    :param data: dataset to filter
    :param criteria: list of filtering criterion names (possible variants are features, target, idx)
    :return: dataset without NaNs
    """

    data_to_clean = copy(data)

    for criterion in criteria:

        datasets_for_criterion = {
            'features': data_to_clean.features,
            'target': data_to_clean.target,
            'idx': data_to_clean.idx
        }
        array_with_nans = datasets_for_criterion[criterion]

        if array_with_nans is not None:
            nans = np.isnan(array_with_nans)
            idx = data_to_clean.idx
            features = data_to_clean.features
            target = data_to_clean.target

            # TODO try to simplify the 1D/2D logic
            # remove all rows with nan in array_with_nans
            if len(array_with_nans.shape) == 1:
                if not nans.any():
                    # if there is no nans at all, go to next criterion
                    continue

                idx = idx[~nans[0:len(idx)]]
                features = features[~nans[0:len(features)]] if data.features is not None else features
                target = target[~nans[0:len(target)]] if data.target is not None else target
            elif len(array_with_nans.shape) == 2:
                if not nans.any().any():
                    # if there is no nans at all, go to next criterion
                    continue
                idx = idx[~nans.any(axis=1)[0:len(idx)]]
                features = features[~nans.any(axis=1)[0:len(features)]] if features is not None else features
                target = target[~nans.any(axis=1)[0:len(target)]] if target is not None else target
            else:
                raise NotImplementedError('Dimensionality not supported')

            data_to_clean.idx = idx
            data_to_clean.features = features
            data_to_clean.target = target

    return data_to_clean
