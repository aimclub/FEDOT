"""H23 synthetic live adapter lifecycle; no supplied graph or fabert data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_startup_contract import prepare_run, save, verify

SIDES = ('prepared-false', 'prepared-true')


def synthetic():
    rng = np.random.RandomState(104753)
    y = np.arange(120) % 3
    x = rng.normal(size=(120, 6)).astype('float32')
    x[:, 0] += 2*y
    x[:, 1] += y == 1
    x[:, 4:] = np.arange(240).reshape(120, 2) % 4
    return x, y


def settings(run, enabled):
    return dict(preset='best_quality', n_jobs=8, cv_folds=3, with_tuning=True,
        parallelization_mode='sequential', final_fit_use_remaining_time=False,
        num_of_generations=1, pop_size=4, _portfolio=False,
        _paid_allrow_tuning=enabled, _paid_policy_observation_dir=str(run/'observation'),
        cache_dir=str(run/'cache'))


def check_predictions(probs, predictions, truth):
    _, y = synthetic()
    assert np.array_equal(np.asarray(truth).ravel(), y[90:])
    assert probs.shape == (30, 3) and np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()
    assert np.max(np.abs(probs.sum(axis=1)-1)) < 1e-6
    assert np.array_equal(np.asarray(predictions).ravel(), np.argmax(probs, axis=1))


def child(dataset, config):
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from fedot.core.composer.gp_composer.gp_composer import GPComposer
    stage = Path(config.contract_stage)
    run = stage/'runs'/config.contract_side
    adapter = load_adapter(stage)
    original = GPComposer.compose_pipeline
    composition = []
    def observed(composer, *args, **kwargs):
        before = time.monotonic()
        result = original(composer, *args, **kwargs)
        composition.append(dict(seconds=time.monotonic()-before,
            generations=[dict(label=g.label, count=len(g)) for g in composer.history.generations]))
        return result
    x, y = synthetic()
    assert np.array_equal(np.asarray(dataset.train.X), x[:90])
    assert np.array_equal(np.asarray(dataset.train.y).ravel(), y[:90])
    assert np.array_equal(np.asarray(dataset.test.X), x[90:])
    with patch.object(GPComposer, 'compose_pipeline', observed):
        response = adapter.run(dataset, config)
    check_predictions(response['probabilities'], response['predictions'], response['truth'])
    with (run/'synthetic-predictions.npz').open('xb') as stream:
        np.savez_compressed(stream, probs=response['probabilities'],
            predictions=response['predictions'], truth=response['truth'])
    save(run/'child.json', dict(composition=composition,
        training_duration=response['training_duration'], predict_duration=response['predict_duration'],
        models_count=response['models_count'],
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']}))
    return response


def parent(stage, side, framework_python):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from amlb.data import DatasetType
    from amlb.utils import Namespace
    from frameworks.shared import caller
    import frameworks.FEDOT as framework
    run = stage/'runs'/side
    x, y = synthetic()
    frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])
    frame[['x4','x5']] = frame[['x4','x5']].astype('uint8')
    encoder = LabelEncoder().fit(y[:90])
    dataset = SimpleNamespace(type=DatasetType.multiclass,
        target=SimpleNamespace(label_encoder=SimpleNamespace(classes=encoder.classes_, transform=encoder.transform)),
        train=SimpleNamespace(X=frame.iloc[:90].copy(), y=pd.DataFrame({'target':y[:90]})),
        test=SimpleNamespace(X=frame.iloc[90:].copy(), y=pd.DataFrame({'target':y[90:]})), release=lambda:None)
    config = Namespace(name='h23-synthetic', type='classification', metric='logloss', cores=8,
        seed=104735, max_runtime_seconds=240, job_timeout_seconds=300,
        contract_stage=str(stage), contract_side=side, output_predictions_file=None,
        framework_params=settings(run, side.endswith('-true')))
    real = caller.run_in_venv
    def launch(caller_file, script_file, **kwargs):
        assert script_file == 'exec.py'
        return real(caller_file, str(stage/'live_adapter_contract.py'), 'child',
            python_exec=framework_python, **kwargs)
    class RetainedDirectory:
        def __init__(self, **kwargs): self.path=tempfile.mkdtemp(dir=run/'transport', **kwargs)
        def __enter__(self): return self.path
        def __exit__(self,*args): return False
    (run/'transport').mkdir()
    before=time.monotonic()
    with patch.object(caller,'run_in_venv',launch), patch.object(caller,'TemporaryDirectory',RetainedDirectory), \
         patch.object(caller,'rconfig',lambda:Namespace(root_dir=str(stage/'harness'),monitoring=Namespace(statistics=[]))):
        response=framework.run_fedot_tabular(dataset,config)
    save(run/'parent.json',dict(seconds=time.monotonic()-before,response=response))


def audit(stage):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.audit_paid_policy_pair import audit_observation
    count=verify(stage)
    results={}
    for side in SIDES:
        run=stage/'runs'/side
        read=lambda name:json.loads((run/name).read_text())
        p=read('process.json');c=read('child.json');parent_result=read('parent.json')
        assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=330
        assert parent_result['seconds']<=320 and c['training_duration']<=240
        assert parent_result['response']['models_count']==c['models_count']>=1
        assert len(c['composition'])==1 and c['composition'][0]['generations']
        assert c['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
        assert c['environment']['PYTHONPATH']==f'{stage}/harness:{stage}/vendor:{stage}/source:{stage}/harness'
        with np.load(run/'synthetic-predictions.npz',allow_pickle=False) as z:
            check_predictions(z['probs'],z['predictions'],z['truth'])
        observation=run/'observation'
        fit=read('observation/fit.json')
        assert fit['error'] is None and fit['policy_enabled']==side.endswith('-true')
        assert abs(fit['api_seconds']-216)<1e-8 and fit['seconds']<=216
        history=OptHistory.load(str(observation/'history.json'))
        assert len(history.generations)>=2
        assert any(g.label=='final_choices' for g in history.generations)
        final=graph_record(PipelineAdapter().restore(history.generations[-1][0].graph))
        assert final==fit['final_graph']
        # Reuse independent CV/partial-probability/history/selection audit where tuning entered.
        detail=audit_observation(observation)
        policy=read('observation/tuning-exit.json')['policy'] if (observation/'tuning-exit.json').exists() else None
        if side.endswith('-true'):
            assert policy and policy['status'] in ['ineligible_original_path','insufficient_or_unsupported_budget','evaluated']
        else: assert policy is None
        dbs=list((run/'cache').glob('cache_*.*_db'))
        assert len(dbs)>=3 and all(p.resolve().parent==(run/'cache').resolve() for p in dbs)
        results[side]=dict(process=p, parent_seconds=parent_result['seconds'],
            composition=c['composition'], policy=policy, observation=detail,
            valid_predictions=True, cache_files=len(dbs))
    return dict(passed=True,verified_files=count,sides=results,
        scope='Synthetic lifecycle correctness with bounded real evolution; no fabert, no quality gate or full-pair retry')


def suite(stage, framework_python, benchmark_python):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    (stage/'runs').mkdir()
    start=time.monotonic()
    for side in SIDES:
        run=prepare_run(stage/'runs',side)
        env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',
                   AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
        before=time.monotonic()
        with (run/'worker.log').open('x') as log:
            child_process=subprocess.Popen([benchmark_python,str(stage/'live_adapter_contract.py'),'parent',
                '--stage',str(stage),'--side',side,'--framework-python',framework_python],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try: code=child_process.wait(timeout=min(330,670-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child_process.pid,signal.SIGTERM)
                try: child_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child_process.pid,signal.SIGKILL);child_process.wait()
                code=child_process.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before))
        assert code==0 and not timed_out
    result=audit(stage);result['stage_seconds']=time.monotonic()-start
    assert result['stage_seconds']<=670
    save(stage/'independent-audit.json',result)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['child','parent','suite','audit'])
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--side',choices=SIDES)
    parser.add_argument('--framework-python')
    parser.add_argument('--benchmark-python')
    args=parser.parse_args()
    if args.mode=='child':
        from frameworks.shared.callee import call_run
        call_run(child)
    elif args.mode=='parent':parent(args.stage,args.side,args.framework_python)
    elif args.mode=='audit':print(json.dumps(audit(args.stage),indent=2))
    else:
        try:suite(args.stage,args.framework_python,args.benchmark_python)
        except BaseException as error:
            save(args.stage/'FAILED',dict(error=repr(error)));raise
