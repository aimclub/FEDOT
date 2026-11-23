"""H8: bounded fresh-cost calibration, fixed proposal, charged equal budgets."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from examples.advanced.diagnostics.cache_cost_probe import probability_loss, sha, worker, write_json

PROPOSAL = {'colsample_bylevel': .5}
POLICY_SECONDS = 600


def calibrated_cap(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid calibration')
    return min(180., 1.3 * seconds)


def trial_fits(remaining, cap):
    return remaining >= 2 * (cap + 30)


def execute_cell(stage, policy, label, cap, params, deadline):
    name = 'fresh_' + policy + '_' + label
    spec = {'cap': cap, 'params': params}
    write_json(stage / 'cells' / (name + '.spec.json'), spec)
    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--stage', str(stage), '--label', name]
    started = time.monotonic()
    external = min(cap + 30, deadline - started)
    if external <= 1:
        return {'name': name, 'valid': False, 'status': 'not_admitted', 'process_seconds': 0.}
    env = os.environ.copy()
    env.pop('PYTEST_CURRENT_TEST', None)
    with (stage / 'cells' / (name + '.log')).open('x') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = p.wait(timeout=external)
            status = 'exited'
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=min(5, max(.1, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            code, status = p.returncode, 'external_timeout'
    record = {'name': name, 'status': status, 'exit_code': code, 'evaluation_cap': cap,
              'external_cap': external, 'process_seconds': time.monotonic() - started, 'valid': False}
    path = stage / 'cells' / (name + '.json')
    if code == 0 and path.exists():
        evaluation = json.loads(path.read_text())
        record.update(valid=evaluation['valid'], evaluation=evaluation)
    write_json(stage / 'cells' / (name + '.process.json'), record)
    return record


def run_policy(stage, policy, reference):
    started = time.monotonic()
    deadline = started + POLICY_SECONDS
    native_loss = reference['independent_losses']['cached_a']
    baseline_cap = 1.3 * max(reference['cached_seconds'])
    initial_cap = baseline_cap if policy == 'baseline' else 180.
    initial = execute_cell(stage, policy, 'initial', initial_cap, {}, deadline)
    if policy == 'candidate' and not initial['valid']:
        result = {'status': 'calibration_failed', 'initial': initial,
                  'policy_seconds': time.monotonic() - started, 'valid_proposals': 0}
        write_json(stage / (policy + '.json'), result)
        return result
    cap = baseline_cap if policy == 'baseline' else calibrated_cap(initial['evaluation']['seconds'])
    selected, selected_loss = {}, native_loss
    proposal = {'valid': False, 'status': 'not_admitted'}
    if trial_fits(deadline - time.monotonic(), cap):
        proposal = execute_cell(stage, policy, 'proposal', cap, PROPOSAL, deadline)
        if proposal['valid'] and proposal['evaluation']['fitness'] < selected_loss:
            selected, selected_loss = PROPOSAL.copy(), proposal['evaluation']['fitness']
    final = execute_cell(stage, policy, 'confirmation', cap, selected, deadline)
    result = {'status': 'evaluated', 'initial': initial, 'proposal': proposal,
              'confirmation': final, 'selected_params': selected, 'selected_cv': selected_loss,
              'next_evaluation_cap': cap, 'policy_seconds': time.monotonic() - started,
              'valid_proposals': int(proposal['valid'])}
    write_json(stage / (policy + '.json'), result)
    return result


def audit(stage):
    reference = json.loads((stage / 'h7-independent-audit.json').read_text())
    with np.load(stage / 'train.npz', allow_pickle=False) as z:
        truth = z['target'].ravel()
    with np.load(stage / 'h7-reference.npz', allow_pickle=False) as z:
        paired = {k: z[k].copy() for k in z.files}
    policies, valid_losses, fitted = {}, {}, {}
    native_graph = reference['cells']['cached_a']['graph_id_before']
    native_params = ast.literal_eval(native_graph.split('/n_catboost_', 1)[1])
    for policy in ('baseline', 'candidate'):
        result = json.loads((stage / (policy + '.json')).read_text())
        policies[policy] = result
        if result['policy_seconds'] > POLICY_SECONDS:
            raise ValueError('Policy budget exceeded')
        process_total = sum(result.get(k, {}).get('process_seconds', 0)
                            for k in ('initial', 'proposal', 'confirmation'))
        if process_total > result['policy_seconds'] + .01:
            raise ValueError('Uncharged evaluation time')
        if result['valid_proposals'] != int(result.get('proposal', {}).get('valid', False)):
            raise ValueError('Invalid proposal count')
        if result['status'] == 'evaluated':
            expected_cap = (1.3 * max(reference['cached_seconds']) if policy == 'baseline'
                            else calibrated_cap(result['initial']['evaluation']['seconds']))
            if result['next_evaluation_cap'] != expected_cap:
                raise ValueError('Calibration cap mismatch')
        for label in ('initial', 'proposal', 'confirmation'):
            cell = result.get(label, {})
            if not cell.get('valid'):
                continue
            name = cell['name']
            path = stage / 'cells' / (name + '.json')
            record = json.loads(path.read_text())
            if record != cell['evaluation'] or cell['process_seconds'] < record['seconds']:
                raise ValueError('Process/evaluation accounting mismatch')
            if (not record['valid'] or len(record['folds']) != 3 or
                    sha(path.with_suffix('.npz')) != record['npz_sha256'] or
                    sha(stage / 'train.npz') != record['data_sha256']):
                raise ValueError('Invalid completed artifact')
            expected_params = PROPOSAL if label == 'proposal' else result.get('selected_params', {}) if label == 'confirmation' else {}
            if record['parameter_override'] != expected_params or record['graph_id_before'] != record['graph_id_after']:
                raise ValueError('Graph/parameter mismatch')
            actual_params = ast.literal_eval(record['graph_id_before'].split('/n_catboost_', 1)[1])
            if actual_params != {**native_params, **expected_params}:
                raise ValueError('Actual graph parameters differ')
            if record['cache_enabled'] or record['cache_before'] or record['cache_after']:
                raise ValueError('Unexpected cache')
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                losses = []
                for key in paired:
                    if not key.startswith('probs_') and not np.array_equal(z[key], paired[key]):
                        raise ValueError('Paired split/classes mismatch')
                for i in range(3):
                    rows = z[f'rows_{i}']
                    if not np.array_equal(z[f'truth_{i}'], truth[rows]):
                        raise ValueError('Truth mismatch')
                    losses.append(probability_loss(z[f'probs_{i}'], z[f'truth_{i}'], z['classes']))
                    if not expected_params and not np.allclose(z[f'probs_{i}'], paired[f'probs_{i}'], atol=1e-8, rtol=0):
                        raise ValueError('Native probabilities differ from common incumbent')
                if not math.isclose(float(np.mean(losses)), record['fitness'], abs_tol=1e-12, rel_tol=0):
                    raise ValueError('Loss mismatch')
            valid_losses[name] = record['fitness']
            fitted[name] = [f['fitted_nodes'] for f in record['folds']]
    a, b = policies['baseline'], policies['candidate']
    final = b.get('confirmation', {})
    passed = (b['valid_proposals'] > a['valid_proposals'] and final.get('valid', False) and
              final['evaluation']['fitness'] <= 1.01 * reference['independent_losses']['cached_a'])
    return {'gate_passed': bool(passed), 'valid_losses': valid_losses, 'fitted_nodes': fitted,
            'policy_seconds': {k: v['policy_seconds'] for k, v in policies.items()},
            'valid_proposals': {k: v['valid_proposals'] for k, v in policies.items()},
            'selected_params': {k: v.get('selected_params') for k, v in policies.items()},
            'next_evaluation_caps': {k: v.get('next_evaluation_cap') for k, v in policies.items()},
            'native_incumbent_cv': reference['independent_losses']['cached_a']}


def suite(stage):
    import fcntl
    with (stage.parents[2] / 'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = (p / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(Path(a.decode(errors='replace')).name in
                   ('runbenchmark.py', 'exec.py', 'cache_cost_probe.py', 'fresh_budget_probe.py',
                    'weak_dataset_probe.py', 'run_followup.py', 'queue_driver.py') for a in args):
                raise RuntimeError('Host occupied: ' + p.name)
        manifest = json.loads((stage / 'manifest.json').read_text())
        assert manifest['preflight_passed']
        for relative, expected in manifest['files'].items():
            assert (stage / relative).resolve().is_relative_to(stage)
            assert sha(stage / relative) == expected, relative
        reference = json.loads((stage / 'h7-independent-audit.json').read_text())
        assert reference['gate_passed']
        write_json(stage / 'STARTED', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid()})
        try:
            for policy in ('baseline', 'candidate'):
                run_policy(stage, policy, reference)
            write_json(stage / 'independent-audit.json', audit(stage))
            write_json(stage / 'COMPLETE', {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        except BaseException as error:
            write_json(stage / 'FAILED', {'error': repr(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'suite', 'audit'))
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--label')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    stage = args.stage.resolve()
    if args.mode == 'worker':
        spec = json.loads((stage / 'cells' / (args.label + '.spec.json')).read_text())
        worker(stage, args.label, args.smoke, spec)
    elif args.mode == 'suite':
        suite(stage)
    else:
        print(json.dumps(audit(stage), allow_nan=False))
