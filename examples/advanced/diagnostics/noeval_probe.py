"""H13: eight tiny real-model correctness cells; no benchmark/quality claim."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split

from examples.advanced.diagnostics.cache_cost_probe import sha, write_json, probability_loss

MODEL = 'fedot/core/operations/evaluation/operation_implementations/models/boostings_implementations.py'


def dataset(kind):
    rng = np.random.RandomState(104739)
    x = rng.normal(size=(300, 12))
    y = np.arange(300) % 7 if kind == 'classification' else x[:, 0]*2-x[:, 1]+rng.normal(scale=.1, size=300)
    train, test = train_test_split(np.arange(300), test_size=60, random_state=104739,
                                   stratify=y if kind == 'classification' else None)
    return x, y, train, test


def audit(stage):
    results = {}
    for kind in ('classification','regression'):
        _, truth, train, test = dataset(kind)
        for use_eval in (True,False):
            for variant in ('baseline','candidate'):
                label=f'{kind}_{int(use_eval)}_{variant}'
                p=stage/'cells'/(label+'.json');r=json.loads(p.read_text())
                assert r['module_sha256']==sha(stage/variant/MODEL)
                assert r['callback_count']==1 and r['thread_count']==8
                assert 0 <= r['seconds'] <= 35
                if variant=='baseline' and not use_eval:
                    assert r['status']=='expected_alias_error'
                    assert all(x in r['error'] for x in ['only one','od_wait','early_stopping_rounds'])
                    results[label]={'status':r['status']}
                    continue
                assert r['status']=='fitted' and r['prediction_sha256']==sha(p.with_suffix('.npz'))
                assert r['pool_rows']==([192,48] if use_eval else [240])
                if not use_eval:
                    assert r['trees']==12
                    assert not set(r['model_params']) & {'od_wait','od_type','od_pval','early_stopping_rounds'}
                    assert r['model_params']['use_best_model'] is False
                with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
                    assert np.array_equal(z['train_rows'], train) and np.array_equal(z['test_rows'], test)
                    assert np.array_equal(z['truth'], truth[test])
                    if kind=='classification':score=probability_loss(z['prediction'],z['truth'],z['classes'])
                    else:
                        assert z['prediction'].shape==z['truth'].shape and np.isfinite(z['prediction']).all()
                        score=float(np.sqrt(np.mean((z['prediction']-z['truth'])**2)))
                assert abs(score-r['score'])<1e-10
                results[label]={'status':'fitted','score':score,'trees':r['trees'],'pool_rows':r['pool_rows']}
            if use_eval:
                a_params = json.loads((stage/'cells'/(f'{kind}_1_baseline.json')).read_text())['model_params']
                b_params = json.loads((stage/'cells'/(f'{kind}_1_candidate.json')).read_text())['model_params']
                assert a_params == b_params
                with np.load(stage/'cells'/(f'{kind}_1_baseline.npz')) as a,np.load(stage/'cells'/(f'{kind}_1_candidate.npz')) as b:
                    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    return dict(gate_passed=True,cells=results,scope='Correctness only; synthetic train/test, no dataset-quality conclusion')


def run(stage):
    import fcntl
    from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        own = str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try: args = [a.decode(errors='replace') for a in p.read_bytes().split(b'\0') if a]
            except OSError: continue
            pid = int(p.parent.name)
            parent = pid == os.getppid() and args and Path(args[0]).name == 'timeout' and own in args
            extra = any(Path(a).name in ('noeval_probe.py', 'adapter_cv_probe.py', 'subspace_validation_probe.py',
                                        'catboost_ceiling_probe.py') for a in args)
            if pid != os.getpid() and not parent and (extra or training_conflict(pid,args,os.getpid(),os.getppid(),own)):
                raise RuntimeError('Host occupied: '+str(pid))
        run_locked(stage)


def run_locked(stage):
    from fedot.core.data.data import InputData
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.operations.operation_parameters import OperationParameters
    manifest=json.loads((stage/'manifest.json').read_text())
    assert manifest['preflight_passed']
    for n,h in manifest['files'].items():assert sha(stage/n)==h,n
    write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    started=time.monotonic()
    try:
        modules={}
        for variant in ('baseline','candidate'):
            p=stage/variant/MODEL
            spec=importlib.util.spec_from_file_location('h13_'+variant,p)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            modules[variant]=module
        for kind in ('classification','regression'):
            x,y,train,test=dataset(kind)
            task=Task(getattr(TaskTypesEnum,kind))
            data=lambda rows:InputData(idx=rows,features=x[rows].copy(),target=y[rows].copy(),task=task,data_type=DataTypesEnum.table)
            operation='catboost' if kind=='classification' else 'catboostreg'
            for use_eval in (True,False):
                for variant in ('baseline','candidate'):
                    label=f'{kind}_{int(use_eval)}_{variant}'
                    params={**PipelineNode(operation).parameters,'num_trees':12,'max_depth':2,'n_jobs':8,
                            'use_eval_set':use_eval,'use_best_model':True,'od_wait':100,'random_seed':104739,
                            'fit_time_limit':30,'allow_writing_files':False}
                    cls=getattr(modules[variant],'FedotCatBoost'+kind.title()+'Implementation')
                    impl=cls(OperationParameters(**params));pool_rows=[]
                    original=impl.convert_to_pool
                    def record_pool(data,identify_cats):
                        pool=original(data,identify_cats);pool_rows.append(pool.num_row());return pool
                    impl.convert_to_pool=record_pool
                    record=dict(module_sha256=sha(stage/variant/MODEL),callback_count=len(impl.fit_callbacks),
                                thread_count=impl.model.get_params()['thread_count'],model_params=impl.model.get_params())
                    before=time.monotonic()
                    try:
                        impl.fit(data(train))
                        prediction=impl.predict_proba(data(test)) if kind=='classification' else impl.predict(data(test)).ravel()
                        classes=np.unique(y) if kind=='classification' else np.array([])
                        score=probability_loss(prediction,y[test],classes) if kind=='classification' else float(np.sqrt(np.mean((prediction-y[test])**2)))
                        p=stage/'cells'/(label+'.npz')
                        with p.open('xb') as f:np.savez_compressed(f,prediction=prediction,truth=y[test],classes=classes,train_rows=train,test_rows=test)
                        record.update(status='fitted',trees=int(impl.model.tree_count_),score=score,prediction_sha256=sha(p))
                    except Exception as error:
                        record.update(status='expected_alias_error' if variant=='baseline' and not use_eval else 'unexpected_error',error=str(error))
                    record.update(seconds=time.monotonic()-before,pool_rows=pool_rows)
                    write_json(stage/'cells'/(label+'.json'),record)
        result=audit(stage);result['stage_seconds']=time.monotonic()-started
        assert result['stage_seconds']<=240
        write_json(stage/'independent-audit.json',result)
        write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    except BaseException as error:
        write_json(stage/'FAILED',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','audit']);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='run':run(a.stage)
    else:print(json.dumps(audit(a.stage)))
