"""Train-only LightGBM ablation; never evaluate the OpenML outer test fold.

Prepare with an OpenML-enabled interpreter, then run with FEDOT's interpreter::

    python multiclass_train_probe.py prepare --cache INPUT --task-id 189355 --fold 0 --output train.npz
    python multiclass_train_probe.py fit --data train.npz --output probe.json

The fit compares only the minimum leaf Hessian, using identical rows, seed,
iteration limit and early stopping. Results are diagnostics, not AMLB scores.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def prepare(args):
    import openml
    from sklearn.preprocessing import LabelEncoder

    set_cache = getattr(openml.config, 'set_root_cache_directory', None)
    (set_cache or openml.config.set_cache_directory)(str(args.cache))
    task = openml.tasks.get_task(args.task_id, download_data=False)
    dataset = task.get_dataset()
    features, target, categorical, _ = dataset.get_data(
        target=task.target_name, dataset_format='array')
    if any(categorical):
        raise ValueError('This numerical-data diagnostic does not encode categorical features')
    train_indices, _ = task.get_train_test_split_indices(fold=args.fold)
    # Discard the outer test before fitting encoders or constructing inner splits.
    features = np.asarray(features[train_indices], dtype=np.float32)
    target = LabelEncoder().fit_transform(np.asarray(target)[train_indices])
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, features=features, target=target,
                            task_id=args.task_id, fold=args.fold)
    print(json.dumps(dict(train_shape=features.shape, classes=len(np.unique(target)))), flush=True)


def fit(args):
    from lightgbm import LGBMClassifier, early_stopping, record_evaluation
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split

    with np.load(args.data) as data:
        features, target = data['features'], data['target']
        task_id, fold = int(data['task_id']), int(data['fold'])
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42)
    results = []
    for weight in (0.001, 1.0):
        history = {}
        model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                               min_child_weight=weight, bagging_fraction=0.85,
                               n_jobs=8, random_state=42, verbosity=-1)
        started = time.monotonic()
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                  callbacks=[early_stopping(30, verbose=False), record_evaluation(history)])
        probabilities = model.predict_proba(x_valid)
        result = dict(min_child_weight=weight, seconds=time.monotonic() - started,
                      best_iteration=model.best_iteration_,
                      inner_logloss=log_loss(y_valid, probabilities, labels=model.classes_),
                      history=history)
        results.append(result)
        print(json.dumps(result), flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(task_id=task_id, fold=fold, split='outer_train_inner_holdout',
                       seed=42, results=results), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    preparation = subparsers.add_parser('prepare')
    preparation.add_argument('--cache', type=Path, required=True)
    preparation.add_argument('--task-id', type=int, required=True)
    preparation.add_argument('--fold', type=int, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    fitting = subparsers.add_parser('fit')
    fitting.add_argument('--data', type=Path, required=True)
    fitting.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    (prepare if arguments.command == 'prepare' else fit)(arguments)
