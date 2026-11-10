import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.advanced.multiclass_train_probe import prepare


@pytest.mark.parametrize('allow,bad_train,expected_error', [
    (False, False, True), (True, False, False), (True, True, True),
])
def test_binary_category_opt_in_uses_outer_train_only(tmp_path, monkeypatch, allow, bad_train, expected_error):
    # Outer test contains a value the validator must never inspect.
    features = np.array([[0, 4], [2 if bad_train else 1, 5], [999, 6]])
    dataset = SimpleNamespace(get_data=lambda **kwargs: (features, np.array(['a', 'b', 'c']),
                                                       [True, False], None))
    task = SimpleNamespace(get_dataset=lambda: dataset, target_name='target',
                           get_train_test_split_indices=lambda **kwargs: (np.array([0, 1]), np.array([2])))
    openml = SimpleNamespace(config=SimpleNamespace(set_root_cache_directory=lambda path: None),
                             tasks=SimpleNamespace(get_task=lambda *args, **kwargs: task))
    monkeypatch.setitem(sys.modules, 'openml', openml)
    args = SimpleNamespace(cache=tmp_path, task_id=7593, fold=3, output=tmp_path / 'train.npz',
                           allow_numeric_binary_categories=allow)
    if expected_error:
        with pytest.raises(ValueError):
            prepare(args)
        assert not args.output.exists()
    else:
        prepare(args)
        with np.load(args.output) as data:
            assert data['features'].shape == (2, 2)
            assert data['features'][:, 0].tolist() == [0, 1]
            assert data['target'].tolist() == [0, 1]
