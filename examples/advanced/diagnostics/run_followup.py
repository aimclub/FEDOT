"""Exclusive single-host queue for a prepared follow-up stage (Linux only)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def verify(stage):
    mapping = json.loads((stage / 'all-source-sha256.json').read_text())
    for relative, expected in mapping.items():
        path = (stage / relative).resolve()
        if not path.is_relative_to(stage.resolve()):
            raise ValueError('Source manifest escapes stage')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed source: ' + relative)
    subprocess.run(['sha256sum', '--check', 'runtime.sha256'], cwd=stage, check=True)


def check_idle():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(arg).name == 'runbenchmark.py' or
               arg.endswith('/frameworks/FEDOT/exec.py') or
               Path(arg).name in ('weak_dataset_probe.py', 'fixed_multiclass_graph_probe.py')
               for arg in args):
            raise RuntimeError('Training process is already active: PID ' + entry.name)


def main(root, stage, mode, after=None):
    stage, root = stage.resolve(), root.resolve()
    if not stage.is_relative_to(root / 'experiments/pr1467-diagnostics-20261109'):
        raise ValueError('Unexpected stage')
    if after:
        after = after.resolve()
        if not after.is_relative_to(root / 'experiments/pr1467-diagnostics-20261109'):
            raise ValueError('Dependency outside diagnostic root')
        with (stage / 'QUEUED').open('x') as stream:
            stream.write(str(after))
    with (root / 'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if after else fcntl.LOCK_NB))
        if after and (not (after / 'QUEUE_FINISHED').exists() or (after / 'QUEUE_FAILED').exists()):
            raise RuntimeError('Dependency has not completed successfully')
        check_idle()
        verify(stage)
        preflight = json.loads((stage / 'preflight-passed.json').read_text())
        for relative, expected in preflight.items():
            if hashlib.sha256((stage / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Preflight log changed')
        with (stage / 'QUEUE_STARTED').open('x') as stream:
            stream.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        try:
            if mode in ('refit', 'horizon'):
                script = stage / 'candidate/examples/advanced/run_multiclass_quality_pair.sh'
                command = ['bash', '-s', '--', str(root), 'large',
                           'Dionis' if mode == 'refit' else 'Covertype', '6', stage.name,
                           'true' if mode == 'refit' else 'false', 'baseline candidate', 'none', 'isolated']
                subprocess.run(command, input=script.read_bytes().replace(b'\r\n', b'\n'),
                               cwd=stage, check=True)
            elif mode == 'category':
                if after is None:
                    raise ValueError('Category screen requires the completed fabert train-only stage')
                env = os.environ.copy()
                env.update(PYTHONPATH=f'{stage}/vendor/candidate:{stage}/baseline',
                           TMPDIR=str(root / 'tmp'), OMP_NUM_THREADS='8',
                           OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8',
                           FEDOT_CATEGORY_CANDIDATE=str(stage / 'candidate/fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py'),
                           FEDOT_DIAGNOSTIC_VENDOR=str(stage / 'vendor/candidate'))
                # Run host-native cheap model contracts only AFTER the previous
                # worker releases the host lock, before any dataset fit.
                with (stage / 'preflight-category.log').open('x') as log:
                    subprocess.run(['timeout', '180s',
                                    str(root / 'automlbenchmark/frameworks/FEDOT/venv/bin/python'),
                                    '-m', 'pytest', '-q',
                                    str(stage / 'candidate/examples/advanced/diagnostics/test_xgb_category_runtime.py'),
                                    str(stage / 'candidate/examples/advanced/diagnostics/test_category_schema.py'),
                                    str(stage / 'candidate/examples/advanced/diagnostics/test_runtime_contracts.py'),
                                    '-k', 'not real_sanitized'],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, check=True, cwd=stage)
                env['PYTHONPATH'] = f'{stage}/vendor/candidate:{stage}/candidate'
                with (stage / 'probe.log').open('x') as log:
                    subprocess.run(['timeout', '--kill-after=30s', '1320s',
                                    str(root / 'automlbenchmark/frameworks/FEDOT/venv/bin/python'),
                                    str(stage / 'candidate/examples/advanced/diagnostics/weak_dataset_probe.py'),
                                    'category', '--data', str(after / 'train.npz'),
                                    '--output', str(stage / 'screen'), '--stage', str(stage)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, check=True, cwd=stage)
            else:
                task_id = {'fabert': 168910, 'Covertype': 7593}[mode]
                env = os.environ.copy()
                code = stage / 'candidate'
                env.update(PYTHONPATH=f'{stage}/vendor/candidate:{code}:{stage}/harness',
                           TMPDIR=str(root / 'tmp'), TMP=str(root / 'tmp'), TEMP=str(root / 'tmp'),
                           JOBLIB_TEMP_FOLDER=str(root / 'tmp'),
                           OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
                env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY', None)
                # Preparation is exclusively from the selected OpenML train fold.
                prepare = [str(root / 'venvs/automlbenchmark/bin/python'),
                           str(code / 'examples/advanced/multiclass_train_probe.py'), 'prepare',
                           '--cache', str(root / 'input'), '--task-id', str(task_id), '--fold', '6',
                           '--output', str(stage / 'train.npz')]
                if mode == 'Covertype':
                    prepare.append('--allow-numeric-binary-categories')
                with (stage / 'prepare.log').open('x') as log:
                    subprocess.run(['timeout', '--kill-after=30s', '1200s', *prepare],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, check=True, cwd=stage)
                with (stage / 'probe.log').open('x') as log:
                    subprocess.run(['timeout', '--kill-after=30s', '5460s',
                                    str(root / 'automlbenchmark/frameworks/FEDOT/venv/bin/python'),
                                    str(code / 'examples/advanced/diagnostics/weak_dataset_probe.py'),
                                    'screen', '--data', str(stage / 'train.npz'), '--output', str(stage / 'screen'),
                                    '--task', mode], stdout=log, stderr=subprocess.STDOUT,
                                   env=env, check=True, cwd=stage)
        except BaseException as error:
            with (stage / 'QUEUE_FAILED').open('x') as stream:
                stream.write(type(error).__name__ + ': ' + str(error))
            raise
        if not (stage / 'QUEUE_FINISHED').exists():
            with (stage / 'QUEUE_FINISHED').open('x') as stream:
                stream.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--mode', choices=['refit', 'horizon', 'fabert', 'Covertype', 'category'], required=True)
    parser.add_argument('--after', type=Path)
    args = parser.parse_args()
    main(args.root, args.stage, args.mode, args.after)
