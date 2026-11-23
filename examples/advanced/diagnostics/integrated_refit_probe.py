"""H19: actual API/composer paid tuner plus full TRAIN refit; controlled graph."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_cost_probe import sha,write_json,probability_loss

PHASES=('initial','proposal','confirmation')


def run_case(stage, smoke=False):
    started=time.monotonic()
    import pandas as pd
    import fedot,golem
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.adapter_fidelity_probe import summarize
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.allrow_fit_probe import instrument_fit

    with np.load(stage/'train.npz',allow_pickle=False) as z:
        x,y=z['features'].copy(),z['target'].ravel().copy()
    frame=pd.read_pickle(stage/'train-frame.pkl')
    classes=np.unique(y)
    adapter=load_adapter(stage);original=adapter.Fedot;captured={};current={}
    config=SimpleNamespace(type='classification',seed=104735,framework_params={'_paid_allrow_tuning':True})
    (stage/'cells').mkdir()

    def phase():
        model=captured['model']
        return PHASES[len(model.api_composer._experimental_paid_policy.record['calls'])]

    original_prepare=PipelineObjectiveEvaluate.prepare_graph
    def prepare(evaluator,graph,train_data,fold_id=None,n_jobs=-1,**kwargs):
        name=phase();pool_dir=stage/'pool-audits'/name
        (pool_dir/'pools').mkdir(parents=True,exist_ok=True)
        previous=instrument_fit(pool_dir)
        current[(name,fold_id)]=np.asarray(train_data.idx).copy()
        try:return original_prepare(evaluator,graph,train_data,fold_id,n_jobs,**kwargs)
        finally:FedotCatBoostImplementation.fit=previous

    def metric(graph,reference_data,fold_id=None,**kwargs):
        name=phase();p=stage/'cells'/f'{name}-fold{fold_id}'
        probs=np.asarray(graph.predict(reference_data).predict)
        truth=np.asarray(reference_data.target).ravel()
        rows=np.asarray(reference_data.idx)
        final=graph.root_node.fitted_operation.model
        with p.with_suffix('.npz').open('xb') as f:
            np.savez_compressed(f,probs=probs,truth=truth,rows=rows,
                                train_rows=current[(name,fold_id)],classes=classes)
        loss=probability_loss(probs,truth,classes)
        write_json(p.with_suffix('.json'),dict(graph=graph_record(graph),loss=loss,
                   trees=int(final.tree_count_),categories=final.get_cat_feature_indices(),
                   npz_sha256=sha(p.with_suffix('.npz'))))
        return loss

    def factory(*args,**kwargs):
        model=original(*args,**kwargs);captured['model']=model
        cat=PipelineNode('catboost',nodes_from=[PipelineNode('scaling')])
        if smoke:cat.parameters={**cat.parameters,'num_trees':12,'max_depth':2}
        pipe=Pipeline(cat);pipe.replace_n_jobs_in_nodes(8)
        model.api_composer.metrics=[metric]
        def assume(data):
            assert np.array_equal(data.idx,np.arange(len(y))) and np.array_equal(np.asarray(data.target).ravel(),y)
            assert np.array_equal(np.asarray(data.features,dtype=np.float32),x)
            write_json(stage/'input.json',dict(schema=summarize(data)))
            return [deepcopy(pipe)],deepcopy(pipe)
        model.api_composer.propose_and_fit_initial_assumption=assume
        model.api_composer.compose_pipeline=lambda data,*args:(deepcopy(pipe),[deepcopy(pipe)],SimpleNamespace(history=OptHistory()))
        original_full=model._train_pipeline_on_full_dataset
        def full_fit(*args,**kwargs):
            (stage/'final-fit/pools').mkdir(parents=True)
            before=time.monotonic();previous=instrument_fit(stage/'final-fit')
            try:return original_full(*args,**kwargs)
            finally:
                captured['refit_seconds']=time.monotonic()-before
                FedotCatBoostImplementation.fit=previous
        model._train_pipeline_on_full_dataset=full_fit
        return model

    with patch.object(adapter,'Fedot',factory):
        model=adapter._make_fedot(config,'neg_log_loss',dict(preset='best_quality',n_jobs=8,
            parallelization_mode='sequential',with_tuning=True,cv_folds=3,
            use_operations_cache=False,use_preprocessing_cache=False,use_predictions_cache=False,
            final_fit_use_remaining_time=False),54,5.4)
    fit_start=time.monotonic()
    with patch.object(PipelineObjectiveEvaluate,'prepare_graph',prepare):
        model.fit(frame,y)
    fit_seconds=time.monotonic()-fit_start
    probs=model.predict_proba(frame,probs_for_all_classes=True)
    final=model.current_pipeline.root_node.fitted_operation.model
    with (stage/'final-train.npz').open('xb') as f:
        np.savez_compressed(f,probs=probs,truth=y,classes=classes,rows=np.arange(len(y)))
    model.history.save(str(stage/'history.json'))
    write_json(stage/'result.json',dict(policy=model.api_composer._experimental_paid_policy.record,
               fit_seconds=fit_seconds,refit_seconds=captured['refit_seconds'],
               api_tuning_seconds=model.api_composer.timer.tuning_spend_time.total_seconds(),
               fit_started_at=model._fit_started_at,api_seconds=model.params.timeout*60,
               final_graph=graph_record(model.current_pipeline),final_trees=int(final.tree_count_),
               final_params=final.get_params(),final_categories=final.get_cat_feature_indices(),
               train_rows=len(y),composition_substituted=True,smoke=smoke,
               worker_seconds=time.monotonic()-started,
               npz_sha256=sha(stage/'final-train.npz'),history_sha256=sha(stage/'history.json'),
               runtime=dict(fedot=fedot.__file__,golem=golem.__file__,python=sys.executable,
                            environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})))


def check_final_constructor(params,expected,selected):
    excluded={'n_jobs','use_eval_set','enable_categorical','callbacks','fit_time_limit'}
    if selected:excluded|={'early_stopping_rounds','od_wait','od_type','od_pval'}
    native={k:v for k,v in expected.items() if k not in excluded}
    native['thread_count']=8
    if selected:native['use_best_model']=False
    assert all(params.get(k)==v for k,v in native.items()), 'Final constructor changed'
    if selected:assert not set(params)&{'early_stopping_rounds','od_wait','od_type','od_pval'}


def audit(stage, smoke=False):
    from sklearn.model_selection import StratifiedKFold
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    r=json.loads((stage/'result.json').read_text());p=r['policy'];b=p['bounds']
    with np.load(stage/'train.npz',allow_pickle=False) as z:y=z['target'].ravel()
    n=len(y);cats=[2,3] if smoke else list(range(715,800));trees=12 if smoke else 3000
    assert n==(30 if smoke else 7413) and r['train_rows']==n and r['composition_substituted'] and r['smoke']==smoke
    assert p['status']=='evaluated' and [c['phase'] for c in p['calls']]==list(PHASES)
    assert all(c['valid'] for c in p['calls']) and p['evaluations_count']==1
    assert r['api_seconds']==3240 and b['api_deadline']==r['fit_started_at']+3240
    assert b['refit_reserve']==324 and b['postfit_reserve']==162 and b['allocation']==1100 and b['initial_cap']==300
    assert p['calls'][0]['cap']==300
    cap=min(300,max(5,1.3*p['calls'][0]['seconds']))
    assert p['calls'][1]['cap']==p['calls'][2]['cap']==cap
    assert p['admission']['admitted'] and p['admission']['remaining']>=2*(cap+50)
    assert p['remaining_after_tuning']>=486
    assert abs(p['charged_tuning_seconds']-r['api_tuning_seconds'])<1e-5
    assert sum(c['seconds'] for c in p['calls'])<=p['charged_tuning_seconds']<=1100
    assert r['refit_seconds']<=324 and r['fit_seconds']<=1450
    assert p['charged_tuning_seconds']+r['refit_seconds']<=r['fit_seconds']<=r['worker_seconds']
    initial=p['initial_graph'];proposal=deepcopy(initial)
    assert len(initial)==2 and initial[0]['operation']=='catboost' and initial[0]['parents']==[1]
    assert initial[1]==dict(operation='scaling',params={},parents=[])
    assert initial[0]['params']['num_trees']==trees and initial[0]['params']['use_eval_set'] is True
    proposal[0]['params']['use_eval_set']=False
    assert [c['graph'] for c in p['calls']]==[initial,proposal,proposal]
    losses={};predictions={}
    for phase,expected in zip(PHASES,[initial,proposal,proposal]):
        losses[phase]=[];predictions[phase]=[]
        reference=np.load(stage/f'reference-{phase}.npz',allow_pickle=False) if not smoke else None
        try:
            for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(y,y)):
                path=stage/'cells'/f'{phase}-fold{i}';v=json.loads(path.with_suffix('.json').read_text())
                assert v['graph']==expected and v['categories']==cats and 1<=v['trees']<=trees
                if phase!='initial':assert v['trees']==trees
                assert v['npz_sha256']==sha(path.with_suffix('.npz'))
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                    assert np.array_equal(z['rows'],valid) and np.array_equal(z['train_rows'],train)
                    assert np.array_equal(z['truth'],y[valid]) and np.array_equal(z['classes'],np.unique(y))
                    loss=probability_loss(z['probs'],y[valid],z['classes']);assert abs(loss-v['loss'])<1e-12
                    if reference is not None:
                        for key in ['rows','train_rows','truth','probs']:assert np.array_equal(z[key],reference[f'{key}_{i}']),key
                    losses[phase].append(loss);predictions[phase].append(z['probs'].copy())
                if phase!='initial':
                    sig=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
                    pool=json.loads((stage/'pool-audits'/phase/'pools'/(sig+'.json')).read_text())
                    assert pool['trees']==trees and pool['callback_count']==1 and len(pool['pools'])==1
                    assert pool['pools'][0]['rows']==train.tolist() and pool['pools'][0]['count']==len(train)
                    assert pool['pools'][0]['categories']==cats
                    check_final_constructor(pool['model_params'],expected[0]['params'],True)
        finally:
            if reference is not None:reference.close()
    cv={name:float(np.mean(values)) for name,values in losses.items()}
    assert all(abs(c['fitness']-cv[c['phase']])<1e-10 for c in p['calls'])
    assert all(np.array_equal(a,b) for a,b in zip(predictions['proposal'],predictions['confirmation']))
    selected=cv['confirmation']<=.99*cv['initial']
    expected=proposal if selected else initial
    assert p['returned_graph']==r['final_graph']==expected
    assert r['final_categories']==cats and 1<=r['final_trees']<=trees
    if selected:assert r['final_trees']==trees
    check_final_constructor(r['final_params'],expected[0]['params'],selected)
    pools=list((stage/'final-fit/pools').glob('*.json'));assert len(pools)==1
    pool=json.loads(pools[0].read_text());assert pool['callback_count']==1 and pool['trees']==r['final_trees']
    assert sorted(sum([v['rows'] for v in pool['pools']],[]))==list(range(n))
    assert all(v['categories']==cats for v in pool['pools'])
    if selected:assert len(pool['pools'])==1 and pool['pools'][0]['count']==n
    assert r['npz_sha256']==sha(stage/'final-train.npz') and r['history_sha256']==sha(stage/'history.json')
    with np.load(stage/'final-train.npz',allow_pickle=False) as z:
        assert np.array_equal(z['rows'],np.arange(n)) and np.array_equal(z['truth'],y) and np.array_equal(z['classes'],np.unique(y))
        train_loss=probability_loss(z['probs'],y,z['classes'])
    h=OptHistory.load(str(stage/'history.json'));adapter=PipelineAdapter()
    assert len(h.generations)==3
    assert [graph_record(adapter.restore(g[0].graph)) for g in h.generations]==[initial,proposal,expected]
    assert [g[0].fitness.value for g in h.generations]==[p['calls'][0]['fitness'],p['calls'][1]['fitness'],p['obtained_metric']]
    if not smoke:
        ref=json.loads((stage/'h17-result.json').read_text());assert initial==ref['initial_graph']
        process=json.loads((stage/'process.json').read_text())
        assert process['exit_code']==0 and r['worker_seconds']<=process['seconds']<=1500
        assert r['runtime']['environment']['PYTHONPATH']==':'.join(str(stage/p) for p in ['vendor','source','harness'])
        assert Path(r['runtime']['fedot']).resolve().is_relative_to((stage/'source').resolve())
        assert Path(r['runtime']['golem']).resolve().is_relative_to((stage/'vendor').resolve())
    assert r['runtime']['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
    return dict(gate_passed=bool(selected and all(a<=1.02*b for a,b in zip(losses['confirmation'],losses['initial']))),
                integration_valid=True,selected_allrow=bool(selected),cv=cv,fold_losses=losses,
                tuning_seconds=r['api_tuning_seconds'],refit_seconds=r['refit_seconds'],fit_seconds=r['fit_seconds'],
                worker_seconds=r['worker_seconds'],refit_reserve=324,final_trees=r['final_trees'],
                final_pool_counts=[v['count'] for v in pool['pools']],full_train_self_loss=train_loss,
                scope='Controlled selected graph; actual API/composer/tuner/full TRAIN refit; no evolution/outer quality or speed claim')


def suite(stage):
    import fcntl
    from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=[v.decode(errors='replace') for v in p.read_bytes().split(b'\0') if v]
            except OSError:continue
            pid=int(p.parent.name);parent=pid==os.getppid() and args and Path(args[0]).name=='timeout' and own in args
            extra=any(Path(v).name in ['composer_budget_probe.py','calibrated_tuner_probe.py','integrated_refit_probe.py'] for v in args)
            if pid!=os.getpid() and not parent and (extra or training_conflict(pid,args,os.getpid(),os.getppid(),own)):
                raise RuntimeError('Host occupied: '+str(pid))
        m=json.loads((stage/'manifest.json').read_text());assert m['preflight_passed']
        for n,h in m['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        start=time.monotonic()
        try:
            with (stage/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,own,'worker','--stage',str(stage)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=proc.wait(timeout=1500)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    code=proc.returncode
            seconds=time.monotonic()-start
            write_json(stage/'process.json',dict(exit_code=code,seconds=seconds,wait_limit=1500,kill_grace=5))
            assert code==0 and seconds<=1500
            a=audit(stage);a['stage_seconds']=time.monotonic()-start;assert a['stage_seconds']<=1540
            write_json(stage/'independent-audit.json',a)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as e:write_json(stage/'FAILED',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','worker','audit']);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='suite':suite(a.stage)
    elif a.mode=='worker':run_case(a.stage)
    else:print(json.dumps(audit(a.stage)))
