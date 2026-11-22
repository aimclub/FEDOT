"""Bounded train-only mechanism screens. Never reads an outer test partition.

Each model/CV evaluation executes in a fresh process with a whole-process
deadline; native library timeouts alone do not bound prediction or cleanup.
Outputs are exclusive, probabilities auditable, failures retained. This is
diagnostic code, not a replacement for an AMLB end-to-end comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 104735
HORIZONS = [(100, .1), (150, .1), (300, .05), (450, 1 / 30)]
PROPOSALS = [dict(colsample_bylevel=.7), dict(colsample_bylevel=.4),
             dict(max_depth=4), dict(num_trees=1500),
             dict(learning_rate=.05, num_trees=1800)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def split_rows(target):
    rows = np.arange(len(target))
    fit, rest = train_test_split(rows, test_size=.4, stratify=target, random_state=SEED)
    select, confirm = train_test_split(rest, test_size=.5, stratify=target[rest], random_state=SEED)
    classes = set(np.unique(target))
    for part in (fit, select, confirm):
        if set(np.unique(target[part])) != classes:
            raise ValueError('A partition lacks a class; do not silently change the split')
    return dict(fit=fit, select=select, confirm=confirm)


def validate_probabilities(probabilities, truth, classes):
    p = np.asarray(probabilities)
    y = np.asarray(truth).ravel()
    classes = np.asarray(classes)
    if len(classes) != len(np.unique(classes)) or not np.array_equal(classes, np.unique(classes)):
        raise ValueError('Classes must be unique and sorted')
    if p.shape != (len(y), len(classes)) or not np.isin(y, classes).all():
        raise ValueError('Probability/class alignment mismatch')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError('Unnormalized probabilities')
    return float(log_loss(y, p, labels=classes))


def pipeline_for(kind, params):
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    scale = PipelineNode('scaling')
    if kind == 'stack':
        learners = [PipelineNode(name, nodes_from=[scale]) for name in ('catboost', 'xgboost', 'lgbm')]
        pipe = Pipeline(PipelineNode('logit', nodes_from=learners))
    else:
        pipe = Pipeline(PipelineNode(kind, nodes_from=[scale]))
    for node in pipe.nodes:
        name = node.operation.operation_type
        if name in ('logit', 'lgbm', 'xgboost'):
            node.parameters = {**node.parameters, 'random_state': SEED}
        elif name == 'catboost':
            node.parameters = {**node.parameters, 'random_seed': SEED}
        if name == kind:
            node.parameters = {**node.parameters, **params}
    return pipe


def as_input(x, y, rows):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    return InputData(idx=np.asarray(rows), features=x[rows].copy(), target=y[rows].copy(),
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)


def fit_diagnostics(pipe):
    result = []
    for node in pipe.nodes:
        implementation = node.fitted_operation
        model = getattr(implementation, 'model', implementation)
        item = dict(operation=node.operation.operation_type, parameters=node.parameters)
        for attr in ('n_estimators_', 'best_iteration_', 'tree_count_'):
            value = getattr(model, attr, None)
            if value is not None:
                item[attr] = int(value)
        if hasattr(model, 'get_booster'):
            item['boosted_rounds'] = int(model.get_booster().num_boosted_rounds())
        result.append(item)
    return result


def worker(data_path, spec_path, output):
    from datetime import timedelta
    with np.load(data_path, allow_pickle=False) as data:
        x, y = data['features'], data['target'].ravel()
    spec = json.loads(spec_path.read_text())
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    classes = np.unique(y)
    train = np.asarray(spec['train'])
    if spec.get('cv'):
        folds = StratifiedKFold(spec['cv'], shuffle=True, random_state=SEED)
        splits = [(train[a], train[b]) for a, b in folds.split(train, y[train])]
    else:
        splits = [(train, None)]
    started = time.monotonic()
    predictions, results = {}, []
    for i, (fit_rows, valid_rows) in enumerate(splits):
        pipe = pipeline_for(spec['kind'], spec.get('params', {}))
        fit_start = time.monotonic()
        pipe.fit(as_input(x, y, fit_rows), n_jobs=8,
                 time_constraint=timedelta(seconds=min(600, spec['cap'])))
        fit_seconds = time.monotonic() - fit_start
        targets = {'cv': valid_rows} if valid_rows is not None else spec['predict']
        metrics = {}
        pred_start = time.monotonic()
        for name, rows in targets.items():
            rows = np.asarray(rows)
            probs = pipe.predict(as_input(x, y, rows), output_mode='full_probs').predict
            metrics[name] = validate_probabilities(probs, y[rows], classes)
            predictions[f'{name}_{i}'] = probs
            predictions[f'{name}_{i}_truth'] = y[rows]
            predictions[f'{name}_{i}_rows'] = rows
        results.append(dict(fit_seconds=fit_seconds, prediction_seconds=time.monotonic() - pred_start,
                            metrics=metrics, nodes=fit_diagnostics(pipe)))
        pipe.unfit()
    with output.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, classes=classes, **predictions)
    write_json(output, dict(seconds=time.monotonic() - started, folds=results,
                            prediction_sha256=digest(output.with_suffix('.npz'))))


def run_cell(data, directory, name, spec, cap, deadline):
    cap = min(float(cap), deadline - time.monotonic())
    if cap <= 1:
        return dict(status='not_started_budget', name=name)
    spec = {**spec, 'cap': cap}
    spec_path, output = directory / (name + '.spec.json'), directory / (name + '.json')
    write_json(spec_path, spec)
    started = time.monotonic()
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--data', str(data),
               '--spec', str(spec_path), '--output', str(output)]
    with (directory / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != 'nt'))
        try:
            code = process.wait(timeout=cap)
            status = 'complete' if code == 0 and output.exists() else 'failed'
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            status = 'timeout'
            code = process.returncode
    result = dict(name=name, status=status, process_seconds=time.monotonic() - started,
                  cap=cap, returncode=code)
    if status == 'complete':
        result['evaluation'] = json.loads(output.read_text())
    write_json(directory / (name + '.status.json'), result)
    print(json.dumps(result), flush=True)
    return result


def cell_metric(cell, name):
    return cell['evaluation']['folds'][0]['metrics'][name]


def timed_cost(cell):
    return sum(row['fit_seconds'] + row['prediction_seconds'] for row in cell['evaluation']['folds'])


def successful(cell):
    return cell.get('status') == 'complete'


def fit_spec(kind, train, partitions, params=None):
    return dict(kind=kind, train=np.asarray(train).tolist(),
                predict={k: np.asarray(v).tolist() for k, v in partitions.items()}, params=params or {})


def screen_horizon(data, directory, parts, deadline):
    cells = []
    for rounds, rate in HORIZONS:
        cells.append(run_cell(data, directory, f'h3_{rounds}',
                              fit_spec('lgbm', parts['fit'], parts,
                                       dict(n_estimators=rounds, learning_rate=rate)), 600, deadline))
    good = [c for c in cells if successful(c)]
    if not good or not successful(cells[0]):
        return dict(status='failed_gate', cells=cells)
    winner = min(good, key=lambda c: (cell_metric(c, 'select'), c['name']))
    passed = cell_metric(winner, 'confirm') <= .99 * cell_metric(cells[0], 'confirm')
    # A conservative linear projection is a screen, not a refit guarantee.
    projected_refit = winner['evaluation']['folds'][0]['fit_seconds'] / .6
    passed = passed and projected_refit < 324
    return dict(status='passed' if passed else 'negative', winner=winner['name'],
                projected_refit_seconds=projected_refit, cells=cells)


def screen_initial(data, directory, parts, deadline):
    cells = {}
    for kind in ('logit', 'lgbm', 'catboost', 'stack'):
        cells[kind] = run_cell(data, directory, 'h2_' + kind,
                              fit_spec(kind, parts['fit'], parts), 600, deadline)
    simple = [c for key, c in cells.items() if key != 'stack' and successful(c)]
    if not simple or not successful(cells['stack']):
        return dict(status='failed_gate', cells=cells)
    best = min(simple, key=lambda c: (cell_metric(c, 'select'), c['name']))
    passed = (timed_cost(best) <= .75 * timed_cost(cells['stack']) and
              cell_metric(best, 'confirm') <= 1.01 * cell_metric(cells['stack'], 'confirm'))
    return dict(status='passed' if passed else 'negative', winner=best['name'], cells=cells)


def screen_probabilities(data, directory, parts, y, initial, deadline):
    from sklearn.linear_model import LogisticRegression
    from fedot.api.experimental.tabular_portfolio import _fit_temperature, _apply_temperature
    classes = np.unique(y)
    base_names = ('catboost', 'lgbm')
    base = initial['cells']
    if not all(successful(base[name]) for name in base_names):
        return dict(status='failed_gate', reason='base_model_failed')
    matrices = {part: [] for part in parts}
    for name in base_names:
        with np.load(directory / ('h2_' + name + '.npz')) as pred:
            for part in parts:
                matrices[part].append(pred[part + '_0'])
    meta = {part: np.hstack(values) for part, values in matrices.items()}
    train_y = y[parts['fit']]
    model = LogisticRegression(C=1, max_iter=1000, random_state=SEED, n_jobs=8)
    started = time.monotonic()
    model.fit(meta['fit'], train_y)
    values = {'insample': {p: model.predict_proba(meta[p]) for p in ('select', 'confirm')}}
    meta_seconds = time.monotonic() - started
    values['blend'] = {p: np.mean(matrices[p], axis=0) for p in ('select', 'confirm')}
    oof = np.full_like(meta['fit'], np.nan)
    cv_cells = []
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(parts['fit'], train_y)):
        for model_id, name in enumerate(base_names):
            cell_name = f'h4_oof_{fold}_{name}'
            cell = run_cell(data, directory, cell_name,
                            fit_spec(name, parts['fit'][tr], {'oof': parts['fit'][va]}), 600, deadline)
            cv_cells.append(cell)
            if not successful(cell):
                continue
            with np.load(directory / (cell_name + '.npz')) as p:
                oof[va, model_id * len(classes):(model_id + 1) * len(classes)] = p['oof_0']
    oof_meta_seconds = None
    if np.isfinite(oof).all():
        started = time.monotonic()
        model.fit(oof, train_y)
        values['oof'] = {p: model.predict_proba(meta[p]) for p in ('select', 'confirm')}
        oof_meta_seconds = time.monotonic() - started
    temperature = _fit_temperature(values['insample']['select'], y[parts['select']], labels=classes)
    values['temperature'] = {p: _apply_temperature(values['insample'][p], temperature)
                             for p in ('select', 'confirm')}
    metrics = {}
    for name, predictions in values.items():
        metrics[name] = {p: validate_probabilities(v, y[parts[p]], classes) for p, v in predictions.items()}
        with (directory / ('h4_' + name + '.npz')).open('xb') as stream:
            np.savez_compressed(stream, classes=classes,
                                select=predictions['select'], select_truth=y[parts['select']],
                                confirm=predictions['confirm'], confirm_truth=y[parts['confirm']])
    costs = {name: sum(timed_cost(base[k]) for k in base_names) + meta_seconds for name in values}
    costs['blend'] -= meta_seconds
    if 'oof' in costs:
        costs['oof'] += sum(timed_cost(c) for c in cv_cells) + oof_meta_seconds - meta_seconds
    eligible = [name for name in ('oof', 'blend', 'temperature') if name in metrics
                and metrics[name]['confirm'] <= .99 * metrics['insample']['confirm']
                and costs[name] < 1800]
    winner = min(eligible, key=lambda k: (metrics[k]['select'], ('oof', 'blend', 'temperature').index(k))) if eligible else None
    return dict(status='passed' if winner else 'negative', winner=winner, metrics=metrics,
                temperature=temperature, seconds=costs, cells=cv_cells)


def cv_metric(cell):
    return float(np.mean([fold['metrics']['cv'] for fold in cell['evaluation']['folds']]))


def screen_tuning(data, directory, parts, deadline):
    train = np.concatenate([parts['fit'], parts['select']])
    policies = {}
    for folds in (3, 2):
        search_start = time.monotonic()
        search_end = min(search_start + 900, deadline)
        initial_spec = dict(kind='catboost', params={}, train=train.tolist(), cv=folds)
        initial = run_cell(data, directory, f'h1_{folds}_initial', initial_spec, 600, search_end)
        if not successful(initial):
            policies[str(folds)] = dict(status='failed_gate', initial=initial)
            continue
        # Process startup and all prediction work are included in fresh cost.
        cap = 1.3 * initial['process_seconds']
        cells = []
        for i, params in enumerate(PROPOSALS):
            if time.monotonic() + cap > search_end:
                break
            cells.append(run_cell(data, directory, f'h1_{folds}_trial{i}',
                                  {**initial_spec, 'params': params}, cap, search_end))
        contenders = [(initial, {})] + [(c, PROPOSALS[i]) for i, c in enumerate(cells) if successful(c)]
        best, params = min(contenders, key=lambda pair: cv_metric(pair[0]))
        confirmation = run_cell(data, directory, f'h1_{folds}_confirm_cv',
                                {**initial_spec, 'params': params, 'cv': 3}, 600, deadline)
        refit = run_cell(data, directory, f'h1_{folds}_refit',
                         fit_spec('catboost', train, {'confirm': parts['confirm']}, params), 600, deadline)
        policies[str(folds)] = dict(initial=initial, cells=cells, evaluation_cap=cap,
                                    search_seconds=sum(c['process_seconds'] for c in [initial, *cells]),
                                    best=best['name'], parameters=params,
                                    confirmation=confirmation, refit=refit,
                                    valid_proposals=sum(successful(c) for c in cells))
    a, b = policies.get('3', {}), policies.get('2', {})
    if not all(successful(p.get('refit', {})) and successful(p.get('confirmation', {})) for p in (a, b)):
        status = 'failed_gate'
    else:
        passed = (b['valid_proposals'] > a['valid_proposals'] and
                  cell_metric(b['refit'], 'confirm') <= 1.01 * cell_metric(a['refit'], 'confirm'))
        status = 'passed' if passed else 'negative'
    return dict(status=status, policies=policies, proposals=PROPOSALS)


def screen(data, output, kind):
    output.mkdir(exist_ok=False)
    (output / 'STARTED').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    with np.load(data, allow_pickle=False) as source:
        y = source['target'].ravel()
        identity = dict(task_id=int(source['task_id']), fold=int(source['fold']))
    if identity['fold'] != 6 or identity['task_id'] != {'fabert': 168910, 'Covertype': 7593}[kind]:
        raise ValueError('Unapproved task/fold')
    parts = split_rows(y)
    with (output / 'splits.npz').open('xb') as stream:
        np.savez_compressed(stream, **parts)
    write_json(output / 'identity.json', dict(**identity, seed=SEED, split='outer_train_only',
                                             data_sha256=digest(data), split_sha256=digest(output / 'splits.npz')))
    deadline = time.monotonic() + 5400
    results = {}
    if kind == 'Covertype':
        results['H3'] = screen_horizon(data, output, parts, deadline)
    else:
        results['H2'] = screen_initial(data, output, parts, deadline)
        write_json(output / 'H2.json', results['H2'])
        results['H4'] = screen_probabilities(data, output, parts, y, results['H2'], deadline)
        write_json(output / 'H4.json', results['H4'])
        results['H1'] = screen_tuning(data, output, parts, deadline)
    write_json(output / 'summary.json', results)
    (output / 'COMPLETE').write_text(digest(output / 'summary.json'))
    print(json.dumps({k: v['status'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'screen'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--task', choices=['fabert', 'Covertype'])
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.data, args.spec, args.output)
    else:
        screen(args.data, args.output, args.task)
