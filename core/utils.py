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
