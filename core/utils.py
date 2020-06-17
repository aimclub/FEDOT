from pathlib import Path
import pandas as pd


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def labels_to_dummy_probs(prediction):
    """Returns converted predictions
    using one-hot probability encoding"""
    df = pd.Series(prediction)
    pred_probas = pd.get_dummies(df).values
    return pred_probas


def check_dimensions_of_features(features):
    if len(features.shape) >= 3:
        num_of_samples = features.shape[1]
        features_2d = features.reshape(num_of_samples, -1)
        return features_2d
    else:
        return features
