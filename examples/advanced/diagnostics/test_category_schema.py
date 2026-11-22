import pickle

import numpy as np
import pandas as pd
import pytest

from examples.advanced.diagnostics.category_schema import CategorySchema


def test_categories_are_train_only_and_known_codes_stay_stable():
    train = pd.DataFrame({'c': pd.Categorical(['a', 'c', 'a']), 'x': [1., 2., 3.]})
    predict = pd.DataFrame({'c': pd.Categorical(['c', 'b', None]), 'x': [4., 5., 6.]})
    schema = CategorySchema().fit(train)
    result = schema.transform(predict)
    assert result['c'].cat.codes.tolist() == [1, -1, -1]
    assert list(result['c'].cat.categories) == ['a', 'c']
    pd.testing.assert_series_equal(result['x'], predict['x'])
    assert predict['c'].cat.codes.tolist() == [1, 0, -1]
    restored = pickle.loads(pickle.dumps(schema))
    pd.testing.assert_frame_equal(restored.transform(predict), result)


def test_schema_rejects_reordered_features_and_changed_categorical_mask():
    frame = pd.DataFrame({'c': pd.Categorical(['a', 'b']), 'x': [1., 2.]})
    schema = CategorySchema().fit(frame)
    with pytest.raises(ValueError):
        schema.transform(frame[['x', 'c']])
    with pytest.raises(ValueError):
        schema.transform(frame.assign(c=[0., 1.]))


def test_numeric_regression_features_are_unchanged():
    frame = pd.DataFrame({'a': [1., np.nan, 3.], 'b': [0, 2, 4]})
    pd.testing.assert_frame_equal(CategorySchema().fit(frame).transform(frame), frame)
