"""Bounded H7 cache/fresh cost experiment; all outputs are new and exclusive."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import log_loss

ORDER = ('prime', 'cached_a', 'fresh_a', 'fresh_b', 'cached_b')
SEED = 104735


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def probability_loss(p, y, classes):
    if not np.array_equal(classes, np.unique(classes)):
        raise ValueError('Classes must be sorted and unique')
    if p.shape != (len(y), len(classes)) or not np.isin(y, classes).all():
        raise ValueError('Class/shape mismatch')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError('Unnormalized probabilities')
    return float(log_loss(y, p, labels=classes))


def cost_gate(cached, fresh, predictions_match, cache_activated):
    valid = (len(cached) == len(fresh) == 2 and
             all(math.isfinite(t) and t > 0 for t in cached + fresh))
    return bool(valid and predictions_match and cache_activated and
                min(fresh) > 1.3 * max(cached))


def worker(stage, label, smoke=False, spec=None, input_data=None):
    from datetime import timedelta
    import fedot
    import golem
    from fedot.core.caching.operations_cache import OperationsCache
    from fedot.core.caching.preprocessing_cache import PreprocessingCache
    from fedot.core.caching.predictions_cache import PredictionsCache
    from fedot.core.data.data import InputData
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
    from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    if not smoke:
        assert Path(fedot.__file__).resolve().is_relative_to(stage / 'source')
        assert Path(golem.__file__).resolve().is_relative_to(stage / 'vendor')
    spec = spec or {}
    random.seed(SEED)
    np.random.seed(SEED)
    with np.load(stage / 'train.npz', allow_pickle=False) as z:
        x, y = z['features'], z['target'].ravel()
    classes = np.unique(y)
    data = InputData(idx=np.arange(len(y)), features=x.copy(), target=y.copy(),
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    if input_data is not None:
        from copy import deepcopy
        if (not np.array_equal(input_data.idx, data.idx) or
                not np.array_equal(np.asarray(input_data.target).ravel(), y) or
                not np.array_equal(np.asarray(input_data.features, dtype=np.float32), x, equal_nan=True)):
            raise ValueError('Injected representation must preserve H7 rows, targets and float32 values')
        data = deepcopy(input_data)
    scale = PipelineNode('scaling')
    cat = PipelineNode('catboost', nodes_from=[scale])
    cat.parameters = {**cat.parameters, **spec.get('params', {})}
    if smoke:
        cat.parameters = {**cat.parameters, 'num_trees': 8, 'max_depth': 2,
                          'use_eval_set': True, 'enable_categorical': False}
    if label == 'prime' and not smoke:
        root = PipelineNode('logit', nodes_from=[cat,
                            PipelineNode('xgboost', nodes_from=[scale]),
                            PipelineNode('lgbm', nodes_from=[scale])])
    else:
        root = cat
    pipe = Pipeline(root)
    pipe.replace_n_jobs_in_nodes(8)
    graph_id = pipe.root_node.descriptive_id
    enabled = not label.startswith('fresh')
    cache = {}
    if enabled:
        for name, cls in [('operations_cache', OperationsCache),
                          ('preprocessing_cache', PreprocessingCache),
                          ('predictions_cache', PredictionsCache)]:
            cache[name] = cls(cache_dir=str(stage / 'cache'), custom_pid='h7', use_stats=True)

    def counters():
        return {k: dict(zip(v._db.get_effectiveness_keys(), v._db.get_effectiveness()))
                for k, v in cache.items()}

    timings, arrays = [], {'classes': classes}

    class TimedEvaluator(PipelineObjectiveEvaluate):
        def prepare_graph(self, graph, train_data, fold_id=None, n_jobs=-1, **kwargs):
            start = time.monotonic()
            result = super().prepare_graph(graph, train_data, fold_id, n_jobs, **kwargs)
            fitted = []
            for node in graph.nodes:
                implementation = node.fitted_operation
                model = getattr(implementation, 'model', implementation)
                item = {'operation': node.operation.operation_type}
                for attr in ('tree_count_', 'best_iteration_', 'n_estimators_'):
                    value = getattr(model, attr, None)
                    if value is not None:
                        item[attr] = int(value)
                if hasattr(model, 'get_cat_feature_indices'):
                    item['cat_feature_indices'] = model.get_cat_feature_indices()
                fitted.append(item)
            timings.append({'fold': fold_id, 'prepare_seconds': time.monotonic() - start,
                            'fitted_nodes': fitted})
            arrays[f'train_rows_{fold_id}'] = np.asarray(train_data.idx).copy()
            return result

    def metric(pipeline, reference_data, predictions_cache=None, fold_id=None, **kwargs):
        start = time.monotonic()
        out = pipeline.predict(reference_data, output_mode='default',
                               predictions_cache=predictions_cache, fold_id=fold_id)
        seconds = time.monotonic() - start
        p = np.asarray(out.predict).copy()
        truth = np.asarray(reference_data.target).ravel().copy()
        loss = probability_loss(p, truth, classes)
        arrays[f'probs_{fold_id}'] = p
        arrays[f'truth_{fold_id}'] = truth
        arrays[f'rows_{fold_id}'] = np.asarray(reference_data.idx).copy()
        timings[-1].update(prediction_seconds=seconds, logloss=loss)
        return loss

    splitter = DataSourceSplitter(cv_folds=3, random_seed=spec.get('split_seed', 42))
    evaluator = TimedEvaluator(MetricsObjective(metric), splitter.build(data),
                               evaluation_time_constraint=timedelta(seconds=spec.get('cap', 600)),
                               eval_n_jobs=8, **cache)
    before = counters()
    node_before = dict(cache['predictions_cache']._db.retrieve_stats()) if enabled else {}
    started = time.monotonic()
    fitness = evaluator.evaluate(pipe)
    seconds = time.monotonic() - started
    after = counters()
    node_after = dict(cache['predictions_cache']._db.retrieve_stats()) if enabled else {}
    target = stage / 'cells' / label
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_json(target.with_suffix('.json'), {
        'label': label, 'valid': bool(fitness.valid), 'seconds': seconds,
        'evaluation_cap': spec.get('cap', 600), 'parameter_override': spec.get('params', {}),
        'split_seed': spec.get('split_seed', 42),
        'fitness': float(fitness.value) if fitness.valid else None,
        'folds': timings, 'cache_enabled': enabled, 'cache_before': before,
        'cache_after': after, 'graph_id_before': graph_id,
        'node_cache_before': node_before, 'node_cache_after': node_after,
        'graph_id_after': pipe.root_node.descriptive_id,
        'data_sha256': sha(stage / 'train.npz'), 'npz_sha256': sha(target.with_suffix('.npz')),
        'runtime': {'fedot': fedot.__file__, 'golem': golem.__file__, 'python': sys.executable}})


def audit(stage):
    with np.load(stage / 'train.npz', allow_pickle=False) as source:
        y = source['target'].ravel()
    cells, results = {}, {}
    expected_data = sha(stage / 'train.npz')
    for label in ORDER:
        path = stage / 'cells' / label
        record = json.loads(path.with_suffix('.json').read_text())
        if not record['valid'] or len(record['folds']) != 3:
            raise ValueError('Incomplete CV: ' + label)
        if record['data_sha256'] != expected_data or record['npz_sha256'] != sha(path.with_suffix('.npz')):
            raise ValueError('Artifact hash mismatch')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        losses, all_rows = [], []
        for i in range(3):
            rows, train = arrays[f'rows_{i}'], arrays[f'train_rows_{i}']
            if (len(np.unique(rows)) != len(rows) or np.intersect1d(rows, train).size or
                    not np.array_equal(np.sort(np.concatenate([rows, train])), np.arange(len(y))) or
                    not np.array_equal(arrays[f'truth_{i}'], y[rows])):
                raise ValueError('Invalid split/truth')
            losses.append(probability_loss(arrays[f'probs_{i}'], arrays[f'truth_{i}'], arrays['classes']))
            all_rows.extend(rows)
        if not np.array_equal(np.sort(all_rows), np.arange(len(y))):
            raise ValueError('CV does not cover input exactly once')
        if not math.isclose(float(np.mean(losses)), record['fitness'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Recomputed metric mismatch')
        cells[label] = arrays
        results[label] = record
    reference = cells['cached_a']
    difference = 0.
    graph = results['cached_a']['graph_id_before']
    for label in ORDER[1:]:
        r, arrays = results[label], cells[label]
        if not (r['graph_id_before'] == r['graph_id_after'] == graph):
            raise ValueError('Graph mismatch')
        for key in reference:
            if key.startswith('probs_'):
                difference = max(difference, float(np.max(np.abs(reference[key] - arrays[key]))))
            elif not np.array_equal(reference[key], arrays[key]):
                raise ValueError('Paired rows/classes mismatch')
    def prediction_hits(label):
        r = results[label]
        before, after = r['node_cache_before'], r['node_cache_after']
        return sum(after.get(f'pred_{graph}_default_{i}', 0) >
                   before.get(f'pred_{graph}_default_{i}', 0) for i in range(3))
    hits = {k: prediction_hits(k) for k in ('cached_a', 'cached_b')}
    cached = [results[k]['seconds'] for k in ('cached_a', 'cached_b')]
    fresh = [results[k]['seconds'] for k in ('fresh_a', 'fresh_b')]
    return {'gate_passed': cost_gate(cached, fresh, difference <= 1e-8, min(hits.values()) >= 3),
            'cached_seconds': cached, 'fresh_seconds': fresh, 'max_abs_probability_difference': difference,
            'prediction_hits': hits, 'fresh_min_over_cached_max': min(fresh) / max(cached),
            'source_data_sha256': expected_data, 'independent_losses': {k: r['fitness'] for k, r in results.items()},
            'cells': results}


def suite(stage, smoke=False):
    import fcntl
    root = stage.parents[2]
    with (root / 'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = (p / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(Path(a.decode(errors='replace')).name in
                   ('runbenchmark.py', 'exec.py', 'weak_dataset_probe.py', 'run_followup.py', 'queue_driver.py')
                   for a in args):
                raise RuntimeError('Host occupied by ' + p.name)
        manifest = json.loads((stage / 'manifest.json').read_text())
        for relative, expected in manifest['files'].items():
            assert (stage / relative).resolve().is_relative_to(stage), relative
            assert sha(stage / relative) == expected, relative
        assert manifest['preflight_passed'] is True
        write_json(stage / 'STARTED', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid()})
        deadline = time.monotonic() + 2400
        try:
            for label in ORDER:
                remaining = min(630, deadline - time.monotonic())
                if remaining <= 1:
                    raise TimeoutError('Stage deadline exhausted')
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--stage', str(stage), '--label', label]
                if smoke:
                    command.append('--smoke')
                with (stage / 'cells' / (label + '.log')).open('x') as log:
                    p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = p.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL)
                            p.wait()
                        raise
                if code:
                    raise RuntimeError('Failed cell: ' + label)
                record = json.loads((stage / 'cells' / (label + '.json')).read_text())
                if not record['valid']:
                    raise RuntimeError('Invalid fitness: ' + label)
            write_json(stage / 'independent-audit.json', audit(stage))
            write_json(stage / 'COMPLETE', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        except BaseException as error:
            write_json(stage / 'FAILED', {'error': repr(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'suite', 'audit'))
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--label', choices=ORDER)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    if args.mode == 'worker':
        worker(stage, args.label, args.smoke)
    elif args.mode == 'suite':
        suite(stage, args.smoke)
    else:
        print(json.dumps(audit(stage), allow_nan=False))
