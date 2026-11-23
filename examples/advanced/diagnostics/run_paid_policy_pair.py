"""H21: one immutable full AMLB fold7 pair. No adaptive use of outer results."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def verify(stage):
    m = json.loads((stage/'manifest.json').read_text())
    assert m['preflight_passed']
    for name, digest in m['files'].items():
        p = stage/name
        assert p.resolve().is_relative_to(stage.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name


def command(root, stage, variant):
    assert variant in ['baseline', 'candidate']
    run = stage/'runs'/variant
    return [str(root/'venvs/automlbenchmark/bin/python'),str(stage/'harness/runbenchmark.py'),
        'FEDOT','small','1h8c','-m','local','-p','1','-s','skip','-i',str(root/'input'),
        '-o',str(run/'results'),'-t','fabert','-f','7','--session',f'fedot.h21-{variant}.fabert.fold7',
        '--logging=console:info,app:debug,root:info','--extra=seed=104735',
        '--extra=f._portfolio=false','--extra=f.parallelization_mode=sequential',
        '--extra=f.n_jobs=8','--extra=f.preset=best_quality','--extra=f.with_tuning=true',
        '--extra=f.final_fit_use_remaining_time=false','--extra=f._save_artifacts=info',
        '--extra=f.logging_level=10',f'--extra=f._paid_allrow_tuning={str(variant=="candidate").lower()}',
        f'--extra=f.cache_dir={run/"cache"}',
        f'--extra=f._paid_policy_observation_dir={run/"observation"}']


def validate_run(run, variant):
    # Only validity/budget is checked while queued; no comparison or choice uses score.
    p = run/'results'/f'fedot.h21-{variant}.fabert.fold7/scores/results.csv'
    with p.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and not rows[0].get('info')
    r = rows[0]
    assert all(math.isfinite(float(r[k])) for k in ['logloss','result'])
    assert all(0 <= float(r[k]) <= 3600 for k in ['duration','training_duration'])
    return dict(valid=True, score_sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main(root, stage):
    import fcntl
    from examples.advanced.diagnostics.run_followup import check_idle
    assert stage.resolve().parent == root.resolve()/'experiments/pr1467-diagnostics-20261109'
    with (root/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_idle();verify(stage)
        assert not (stage/'STARTED').exists()
        save(stage/'STARTED', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        before = time.monotonic()
        for variant in ['baseline','candidate']:
            verify(stage)
            run = stage/'runs'/variant
            run.mkdir(parents=True,exist_ok=False)
            env = os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
            cache = root.parent/'.cache/official-1h8c'
            env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',
                AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1',
                TMPDIR=str(root/'tmp'),TMP=str(root/'tmp'),TEMP=str(root/'tmp'),
                JOBLIB_TEMP_FOLDER=str(root/'tmp'),XDG_CACHE_HOME=str(cache),
                XDG_CONFIG_HOME=str(cache/'openml-config'),MPLCONFIGDIR=str(cache/'matplotlib'),
                NUMBA_CACHE_DIR=str(cache/'numba'))
            for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
                env[name]='8'
            save(run/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),command=command(root,stage,variant)))
            start = time.monotonic()
            with (run/'run.log').open('x') as log:
                p = subprocess.Popen(command(root,stage,variant),env=env,cwd=stage,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                timed_out=False
                try:code=p.wait(timeout=3900)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    code=p.returncode
            save(run/'process.json',dict(exit_code=code,seconds=time.monotonic()-start,
                timed_out=timed_out,wait_limit=3900,kill_grace=5))
            try:
                assert code==0 and not timed_out
                save(run/'COMPLETE',validate_run(run,variant))
            except Exception as exc:
                save(run/'FAILED',dict(error=repr(exc)))
            # No metric-dependent admission. If descendants remain, do not overlap them.
            check_idle()
        save(stage/'QUEUE_FINISHED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            seconds=time.monotonic()-before))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    args=p.parse_args()
    try:main(args.root,args.stage)
    except BaseException as exc:
        failure=args.stage/'QUEUE_FAILED'
        if not failure.exists():save(failure,dict(error=repr(exc)))
        raise
