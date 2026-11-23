"""H22: bounded constructor-only missing/prepared custom-cache contract."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


CELLS = ('missing-false', 'prepared-false', 'missing-true', 'prepared-true')


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare_run(parent, name):
    """Exclusive new run/cache, refusing existing paths and linked parents."""
    parent = Path(parent)
    if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent.absolute():
        raise ValueError('parent must be an existing resolved directory')
    if name not in CELLS:
        raise ValueError('unknown fixed cell')
    run = parent/name
    run.mkdir(exist_ok=False)
    (run/'cache').mkdir(exist_ok=False)
    return run


def verify(stage):
    m = json.loads((stage/'manifest.json').read_text())
    for name, digest in m['files'].items():
        p = stage/name
        assert p.resolve().is_relative_to(stage.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
    return len(m['files'])


def worker(stage, name):
    assert name in CELLS
    run = stage/'cells'/name
    from fedot.api.main import Fedot
    from fedot.api.api_utils.api_composer import ApiComposer
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    path = stage/'harness/frameworks/FEDOT/exec.py'
    spec = importlib.util.spec_from_file_location('h22_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    enabled = name.endswith('-true')
    counters = dict(fit=0, compose=0, pipeline_fit=0, evaluate=0)
    def forbid(key):
        def blocked(*args, **kwargs):
            counters[key] += 1
            raise AssertionError('Forbidden training/evaluation: '+key)
        return blocked
    config = SimpleNamespace(type='classification', seed=104735, framework_params={
        '_paid_allrow_tuning': enabled, '_paid_policy_observation_dir': str(run/'observation')})
    params = dict(n_jobs=8, cv_folds=3, preset='best_quality', with_tuning=True,
        parallelization_mode='sequential', final_fit_use_remaining_time=False,
        cache_dir=str(run/'cache'))
    before = time.monotonic()
    with ExitStack() as stack:
        for obj, method, key in [(Fedot,'fit','fit'),(ApiComposer,'obtain_model','compose'),
                (Pipeline,'fit','pipeline_fit'),(PipelineObjectiveEvaluate,'evaluate','evaluate')]:
            stack.enter_context(patch.object(obj, method, forbid(key)))
        try:
            model = adapter._make_fedot(config, 'neg_log_loss', params, 54, 5.4)
        except FileNotFoundError as error:
            assert name.startswith('missing-') and error.filename == str(run/'cache')
            record = dict(status='expected_missing_cache', filename=error.filename)
        else:
            assert name.startswith('prepared-')
            assert callable(getattr(model.api_composer, '_experimental_paid_policy', None)) == enabled
            caches = {}
            for key in ['operations_cache','preprocessing_cache','predictions_cache']:
                cache = getattr(model.api_composer, key)
                assert cache is not None
                db = cache._db.db_path
                assert db.resolve().parent == (run/'cache').resolve()
                with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                caches[key] = dict(path=str(db), tables=tables, entries=len(cache))
                assert len(cache) == 0
            assert (run/'observation').is_dir() and not list((run/'observation').iterdir())
            record = dict(status='constructed', caches=caches, api_seconds=model.params.timeout*60,
                fedot_file=__import__('fedot').__file__, cache_dir=model.params.get('cache_dir'),
                policy_installed=enabled)
    assert not any(counters.values())
    record.update(name=name, enabled=enabled, counters=counters, seconds=time.monotonic()-before,
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(run/'result.json', record)


def audit(stage):
    count = verify(stage)
    results = {}
    for name in CELLS:
        run = stage/'cells'/name
        r = json.loads((run/'result.json').read_text())
        p = json.loads((run/'process.json').read_text())
        assert p['exit_code'] == 0 and not p['timed_out'] and p['seconds'] <= 60
        assert r['name'] == name and r['enabled'] == name.endswith('-true')
        assert not any(r['counters'].values())
        assert r['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
        assert r['environment']['PYTHONPATH'] == f'{stage}/vendor:{stage}/source:{stage}/harness'
        if name.startswith('missing-'):
            assert r['status'] == 'expected_missing_cache'
            assert r['filename'] == str(run/'cache') and not (run/'cache').exists()
            assert not (run/'observation').exists()
        else:
            assert r['status'] == 'constructed' and r['api_seconds'] == 3240
            assert r['policy_installed'] == r['enabled']
            assert Path(r['fedot_file']).is_relative_to(stage/'source')
            assert len(r['caches']) == 3
            for c in r['caches'].values():
                db = Path(c['path'])
                assert db.resolve().parent == (run/'cache').resolve()
                with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    tables=[row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                    assert tables == c['tables']
                    for table in tables:
                        assert table.replace('_','').isalnum()
                        if table != 'effectiveness':
                            assert conn.execute('SELECT COUNT(*) FROM "'+table+'"').fetchone()[0] == 0
                assert c['entries'] == 0
            assert not list((run/'observation').iterdir())
        results[name] = dict(status=r['status'], counters=r['counters'], process=p)
    return dict(verified_files=count, cells=results, passed=True,
                scope='Constructor/cache contract only; zero training; no full-pair retry admission')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    before = time.monotonic()
    (stage/'cells').mkdir()
    for name in CELLS:
        if name.startswith('prepared-'):
            run = prepare_run(stage/'cells', name)
        else:
            run = stage/'cells'/name
            run.mkdir()
        env = os.environ.copy()
        env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',
                   AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[k]='8'
        start = time.monotonic()
        with (run/'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable,str(stage/'cache_startup_contract.py'),
                'worker','--stage',str(stage),'--name',name],env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=min(60,270-(time.monotonic()-before)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-start))
        assert code == 0 and not timed_out
    result = audit(stage)
    result['stage_seconds'] = time.monotonic()-before
    assert result['stage_seconds'] <= 270
    save(stage/'independent-audit.json', result)
    save(stage/'COMPLETE', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['suite','worker','audit'])
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--name',choices=CELLS)
    args=parser.parse_args()
    if args.mode=='worker': worker(args.stage,args.name)
    elif args.mode=='audit': print(json.dumps(audit(args.stage),indent=2))
    else:
        try: suite(args.stage)
        except BaseException as error:
            save(args.stage/'FAILED',dict(error=repr(error)))
            raise
