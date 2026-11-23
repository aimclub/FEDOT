"""H18: synthetic real adapter -> ApiComposer -> paid tuner -> final refit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_cost_probe import sha,write_json,probability_loss


def run_case(stage, smoke=False):
    import pandas as pd
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.allrow_fit_probe import instrument_fit
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    adapter=load_adapter(stage)
    original=adapter.Fedot
    captured={}
    rng=np.random.RandomState(104735)
    n=30 if smoke else 90
    x=rng.normal(size=(n,4));x[:,2:]=np.arange(n*2).reshape(n,2)%3
    y=(np.arange(n)%3).astype(int)
    frame=pd.DataFrame(x,columns=['a','b','c','d'])
    config=SimpleNamespace(type='classification',metric='logloss',cores=8,seed=104735,
                           max_runtime_seconds=360,framework_params={'_paid_allrow_tuning':True,
                           '_portfolio':False,'preset':'best_quality','with_tuning':True,
                           'use_operations_cache':False,'use_preprocessing_cache':False,'use_predictions_cache':False,
                           'final_fit_use_remaining_time':False,'cv_folds':3})
    def factory(*args,**kwargs):
        model=original(*args,**kwargs);captured['model']=model
        scale=PipelineNode('scaling');cat=PipelineNode('catboost',nodes_from=[scale])
        cat.parameters={**cat.parameters,'num_trees':12,'max_depth':2}
        pipe=Pipeline(cat);pipe.replace_n_jobs_in_nodes(8)
        # Controlled selected graph, not a claim that tiny evolution selects it.
        # Real obtain_model, init_params, tune_final_pipeline and final fit remain.
        model.api_composer.propose_and_fit_initial_assumption=lambda data:([deepcopy(pipe)],deepcopy(pipe))
        model.api_composer.compose_pipeline=lambda data,*args:(deepcopy(pipe),[deepcopy(pipe)],SimpleNamespace(history=OptHistory()))
        original_full=model._train_pipeline_on_full_dataset
        def final_fit(*args,**kwargs):
            (stage/'final-fit/pools').mkdir(parents=True)
            previous=instrument_fit(stage/'final-fit')
            try:return original_full(*args,**kwargs)
            finally:FedotCatBoostImplementation.fit=previous
        model._train_pipeline_on_full_dataset=final_fit
        return model
    with patch.object(adapter,'Fedot',factory):
        model=adapter._make_fedot(config,'neg_log_loss',dict(preset='best_quality',n_jobs=8,
            parallelization_mode='sequential',with_tuning=True,cv_folds=3,
            use_operations_cache=False,use_preprocessing_cache=False,use_predictions_cache=False,
            final_fit_use_remaining_time=False),324/60,32.4/60)
    assert model is captured['model']
    # Use actual Fedot.fit clock, API preprocessing, real composer and final refit.
    before=time.monotonic()
    model.fit(frame,y)
    elapsed=time.monotonic()-before
    probs=model.predict_proba(frame,probs_for_all_classes=True)
    policy=model.api_composer._experimental_paid_policy
    final=model.current_pipeline.root_node.fitted_operation.model
    np.savez_compressed(stage/'probabilities.npz',probs=probs,truth=y,classes=np.unique(y))
    model.history.save(str(stage/'history.json'))
    write_json(stage/'result.json',dict(policy=policy.record,fit_seconds=elapsed,
               api_tuning_seconds=model.api_composer.timer.tuning_spend_time.total_seconds(),
               fit_started_at=model._fit_started_at,api_seconds=model.params.timeout*60,
               final_graph=graph_record(model.current_pipeline),final_trees=int(final.tree_count_),
               final_params=final.get_params(),final_categories=final.get_cat_feature_indices(),
               train_rows=len(y),synthetic_only=True,composition_substituted=True,smoke=smoke,
               npz_sha256=sha(stage/'probabilities.npz'),history_sha256=sha(stage/'history.json')))


def audit(stage):
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    r=json.loads((stage/'result.json').read_text());p=r['policy']
    n=30 if r['smoke'] else 90
    assert r['synthetic_only'] and r['composition_substituted'] and r['train_rows']==n
    assert p['status']=='evaluated' and [v['phase'] for v in p['calls']]==['initial','proposal','confirmation']
    assert all(v['valid'] for v in p['calls']) and p['evaluations_count']==1
    b=p['bounds'];assert b['api_deadline']==r['fit_started_at']+r['api_seconds']
    assert b['refit_reserve']==32.4 and abs(b['postfit_reserve']-16.2)<1e-9
    assert abs(b['allocation']-min(1100,b['remaining']-32.4-16.2))<1e-8
    assert b['initial_cap']==min(300.,(b['allocation']-100.)/3.)
    assert p['calls'][0]['cap']==b['initial_cap']
    cap=min(b['initial_cap'],max(5.,1.3*p['calls'][0]['seconds']))
    assert p['calls'][1]['cap']==p['calls'][2]['cap']==cap
    assert p['admission']['admitted'] and p['admission']['remaining']>=2*(cap+50)
    assert p['remaining_after_tuning']>=b['refit_reserve']+b['postfit_reserve']
    assert abs(r['api_tuning_seconds']-p['charged_tuning_seconds'])<1e-5
    assert sum(v['seconds'] for v in p['calls'])<=p['charged_tuning_seconds']<=r['fit_seconds']<240
    initial=p['initial_graph'];proposal=deepcopy(initial)
    cat=next(n for n in proposal if n['operation']=='catboost');cat['params']['use_eval_set']=False
    assert [v['graph'] for v in p['calls']]==[initial,proposal,proposal]
    selected=p['calls'][2]['fitness']<=.99*p['calls'][0]['fitness']
    assert p['returned_graph']==r['final_graph']==(proposal if selected else initial)
    assert r['final_categories']==[2,3] and 1<=r['final_trees']<=12
    if selected:
        assert r['final_trees']==12 and r['final_params']['use_best_model'] is False
        assert not set(r['final_params'])&{'early_stopping_rounds','od_wait','od_type','od_pval'}
    assert r['final_params']['thread_count']==8
    pool_files=list((stage/'final-fit/pools').glob('*.json'));assert len(pool_files)==1
    final_pool=json.loads(pool_files[0].read_text());assert final_pool['callback_count']==1
    assert sorted(sum([v['rows'] for v in final_pool['pools']],[]))==list(range(n))
    if selected:assert len(final_pool['pools'])==1 and final_pool['pools'][0]['count']==n
    else:assert [v['count'] for v in final_pool['pools']]==[int(.8*n),int(.2*n)]
    assert r['npz_sha256']==sha(stage/'probabilities.npz') and r['history_sha256']==sha(stage/'history.json')
    h=OptHistory.load(str(stage/'history.json'));adapter=PipelineAdapter()
    assert len(h.generations)==3
    assert [graph_record(adapter.restore(g[0].graph)) for g in h.generations]==[initial,proposal,r['final_graph']]
    assert [g[0].fitness.value for g in h.generations]==[p['calls'][0]['fitness'],p['calls'][1]['fitness'],p['obtained_metric']]
    with np.load(stage/'probabilities.npz',allow_pickle=False) as z:
        assert z['probs'].shape==(n,3)
        loss=probability_loss(z['probs'],z['truth'],z['classes'])
    return dict(gate_passed=True,fit_seconds=r['fit_seconds'],tuning_seconds=r['api_tuning_seconds'],
                allocation=b['allocation'],refit_reserve=b['refit_reserve'],postfit_reserve=b['postfit_reserve'],
                selected_allrow=selected,final_trees=r['final_trees'],synthetic_train_loss=loss,
                scope='Synthetic adapter factory + actual Fedot.fit/composer/tuner/final refit; selected graph supplied; no fabert or outer result')


def suite(stage):
    import fcntl,os
    from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=[v.decode(errors='replace') for v in p.read_bytes().split(b'\0') if v]
            except OSError:continue
            pid=int(p.parent.name);parent=pid==os.getppid() and args and Path(args[0]).name=='timeout' and own in args
            extra=any(Path(v).name in ['composer_budget_probe.py','calibrated_tuner_probe.py'] for v in args)
            if pid!=os.getpid() and not parent and (extra or training_conflict(pid,args,os.getpid(),os.getppid(),own)):
                raise RuntimeError('Host occupied: '+str(pid))
        m=json.loads((stage/'manifest.json').read_text());assert m['preflight_passed']
        for n,h in m['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        start=time.monotonic()
        try:
            run_case(stage)
            a=audit(stage);a['stage_seconds']=time.monotonic()-start;assert a['stage_seconds']<=300
            write_json(stage/'independent-audit.json',a)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as e:write_json(stage/'FAILED',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','audit']);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='suite':suite(a.stage)
    else:print(json.dumps(audit(a.stage)))
