"""Real FEDOT XGBoost wrapper checks against an explicitly isolated candidate."""
import importlib.util
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.evaluation.operation_implementations.models import boostings_implementations as baseline
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture(scope='module')
def candidate():
    path = os.environ.get('FEDOT_CATEGORY_CANDIDATE')
    if not path:
        pytest.skip('Requires a new isolated candidate module; no imported module is patched')
    assert Path(path).resolve() != Path(baseline.__file__).resolve()
    spec = importlib.util.spec_from_file_location('_category_candidate_runtime', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def data(categories=None, regression=False):
    n = 90 if categories is None else len(categories)
    x = np.column_stack([np.linspace(-1, 1, n), np.arange(n) % 4])
    task = Task(TaskTypesEnum.regression if regression else TaskTypesEnum.classification)
    target = np.sin(x[:, 0]) if regression else np.arange(n) % 3
    if categories is not None:
        x = x.astype(object)
        x[:, 0] = categories
    return InputData.from_numpy(features_array=x, target_array=target, task=task,
                                categorical_idx=[0] if categories is not None else [])


def model(module, categorical=True, evaluation=False, regression=False):
    params = OperationParameters(enable_categorical=categorical, use_eval_set=evaluation,
                                 n_estimators=8, max_depth=2, learning_rate=.1, n_jobs=1,
                                 tree_method='hist', verbosity=0, random_state=104735)
    cls = module.FedotXGBoostRegressionImplementation if regression else module.FedotXGBoostClassificationImplementation
    return cls(params)


def test_unknown_prediction_reproduces_and_candidate_survives_serialization(candidate):
    # Category determines the target: force the tree to actually use this
    # feature. An unused categorical column need not exercise recoding.
    train = data(['a', 'c', 'e'] * 30)
    predict = data(['a', 'b', 'e'] * 10)
    old, new = model(baseline), model(candidate)
    old.fit(train)
    with pytest.raises(Exception, match='categor|Categor'):
        old.predict_proba(predict)
    new.fit(train)
    probs = new.predict_proba(predict)
    assert probs.shape == (30, 3) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1)
    restored = pickle.loads(pickle.dumps(new))
    np.testing.assert_array_equal(probs, restored.predict_proba(predict))
    assert new.predict(predict).shape == (30,)


def test_eval_set_categories_do_not_enter_training_vocabulary(candidate):
    original = data(['a', 'c'] * 45)
    _, valid = train_test_data_setup(original)
    categories = np.array(['a', 'c'] * 45, dtype=object)
    categories[int(valid.idx[0])] = 'eval_only'
    train = data(categories)
    new = model(candidate, evaluation=True)
    new.fit(train)
    assert all('eval_only' not in dtype.categories for dtype in new._category_schema.types.values())
    assert np.isfinite(new.predict_proba(train)).all()


@pytest.mark.parametrize('regression', [False, True])
def test_numerical_models_match_baseline(candidate, regression):
    train = data(regression=regression)
    old = model(baseline, categorical=False, regression=regression)
    new = model(candidate, categorical=False, regression=regression)
    old.fit(train)
    new.fit(train)
    np.testing.assert_allclose(old.predict(train), new.predict(train), rtol=0, atol=0)
