"""Experimental train-fitted categorical vocabulary; no global monkeypatch.

Applied only by the companion patch in a new isolated XGBoost implementation.
Numeric inputs and other model implementations are left alone.
"""
import pandas as pd


class CategorySchema:
    def fit(self, frame):
        self.columns = tuple(frame.columns)
        self.types = {name: frame[name].dtype for name in frame.columns
                      if isinstance(frame[name].dtype, pd.CategoricalDtype)}
        return self

    def transform(self, frame):
        if tuple(frame.columns) != self.columns:
            raise ValueError('Feature order differs from the training schema')
        actual = {name for name in frame if isinstance(frame[name].dtype, pd.CategoricalDtype)}
        if actual != set(self.types):
            raise ValueError('Categorical mask differs from the training schema')
        result = frame.copy()
        for name, dtype in self.types.items():
            # Casting via object is deliberate: unknown categories become NaN,
            # while known category codes refer to the TRAIN vocabulary.
            result[name] = pd.Categorical(frame[name].astype(object), dtype=dtype)
        return result
