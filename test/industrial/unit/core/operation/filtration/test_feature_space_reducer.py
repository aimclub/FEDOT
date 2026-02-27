import numpy as np
import pandas as pd

from fedot_ind.core.operation.filtration.feature_filtration import FeatureSpaceReducer

N_FEATURES = 10
N_SAMPLES = 10


def get_features(add_stable: bool = False):
    feature_dict = {'feature_0': np.random.rand(10),
                    'feature_1': np.random.rand(10)}
    for i in range(2, N_FEATURES):
        feature_dict[f'feature_{i}'] = i * \
            feature_dict[np.random.choice(['feature_0', 'feature_1'])]
    if add_stable:
        last_name = list(feature_dict.keys())[-1]
        feature_dict[last_name] = np.ones(10)
    return pd.DataFrame(feature_dict).values


def test_reduce_feature_space():
    features = get_features()
    cls = FeatureSpaceReducer()
    result = cls.reduce_feature_space(features=features)
    assert result is not None


def test_reduce_feature_space_stable():
    features = get_features(add_stable=True)
    cls = FeatureSpaceReducer()
    result = cls.reduce_feature_space(features=features)
    assert result is not None


def test__drop_correlated_features():
    features = get_features(add_stable=True)
    cls = FeatureSpaceReducer()
    result = cls._drop_correlated_features(corr_threshold=0.99, features=features)
    assert result is not None


def test__drop_stable_features():
    features = get_features(add_stable=True)
    cls = FeatureSpaceReducer()
    result = cls._drop_constant_features(var_threshold=0.99, features=features)
    assert result is not None
