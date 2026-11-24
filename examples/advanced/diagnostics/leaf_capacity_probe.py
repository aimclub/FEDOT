"""H30: one LGBM leaf-size intervention against retained H29 TRAIN CV artifacts."""
import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from examples.advanced.diagnostics.cache_startup_contract import save, verify
from examples.advanced.diagnostics.cache_cost_probe import probability_loss


def candidate_graph(control):
    candidate=deepcopy(control)
    assert len(candidate)==2 and candidate[0]['operation']=='lgbm' and candidate[1]['operation']=='scaling'
    assert candidate[0]['params']['min_data_in_leaf']==29
    candidate[0]['params']['min_data_in_leaf']=20
    return candidate


def gate(control, candidate, active):
    return bool(len(control)==len(candidate)==len(active)==3 and all(active)
        and np.isfinite(control+candidate).all()
        and np.mean(candidate)<=.99*np.mean(control)
        and all(c<=1.02*b for b,c in zip(control,candidate)))


def capture_data(stage,frame,y):
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.adapter_fidelity_probe import CapturedBeforeSearch
    from fedot.core.pipelines.pipeline import Pipeline
    adapter=load_adapter(stage)
    model=adapter._make_fedot(SimpleNamespace(type='classification',seed=104735,framework_params={}),
        'neg_log_loss',dict(preset='best_quality',n_jobs=8,cv_folds=3,with_tuning=True,
        use_operations_cache=False,use_preprocessing_cache=False,use_predictions_cache=False,
        final_fit_use_remaining_time=False),3.6,.36)
    captured={}
    def stop(data):captured['data']=data;raise CapturedBeforeSearch()
    model.api_composer.obtain_model=stop
    with patch.object(Pipeline,'fit',side_effect=AssertionError('No fitting during data capture')):
        try:model.fit(frame,y)
        except CapturedBeforeSearch:pass
    return captured['data']


def worker(stage,output,smoke=False):
    import pandas as pd
    from lightgbm import LGBMClassifier
    from fedot.core.operations.evaluation.operation_implementations.models import boostings_implementations as boosting
    from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
    from examples.advanced.diagnostics.fallback_rng_contract import graph_from_record
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    assert LGBMClassifier().get_params()['min_child_samples']==20
    if smoke:
        x=np.random.RandomState(104764).normal(size=(60,6));y=np.arange(60)%3
        frame=pd.DataFrame(x,columns=[f'x{i}' for i in range(6)])
        graph=[dict(operation='lgbm',params=dict(boosting_type='gbdt',max_depth=-1,
            bagging_fraction=.85,extra_trees=False,min_data_in_leaf=29,n_estimators=8,n_jobs=8,
            use_eval_set=True,enable_categorical=True,early_stopping_rounds=3,verbose=-1),parents=[1]),
            dict(operation='scaling',params={},parents=[])]
    else:
        frame=pd.read_parquet(stage/'train-frame.pd');y=np.load(stage/'train-target.npy',allow_pickle=False).ravel()
        graph=json.loads((stage/'control/evaluation.json').read_text())['graph']
        expected=np.random.RandomState(104763).normal(size=(90,6)).astype('float32')
        labels=np.arange(90)%3;expected[:,0]+=2*labels;expected[:,1]+=labels==1
        expected[:,4:]=np.arange(180).reshape(90,2)%4
        assert np.array_equal(frame.to_numpy(),expected) and np.array_equal(y,labels)
        assert list(frame.columns)==[f'x{i}' for i in range(6)]
        assert [str(v) for v in frame.dtypes]==['float32']*4+['uint8']*2
    data=capture_data(stage,frame,y)
    assert np.array_equal(np.asarray(data.target).ravel(),y) and np.array_equal(data.idx,np.arange(len(y)))
    candidate=candidate_graph(graph);pipe=graph_from_record(candidate)
    splitter=DataSourceSplitter(cv_folds=3,random_seed=42)
    arrays={'target':y,'classes':np.unique(y)};state={};fits=[]
    for fold,(train,valid) in enumerate(splitter.build(data)()):
        arrays[f'train_rows_{fold}']=np.asarray(train.idx);arrays[f'rows_{fold}']=np.asarray(valid.idx)
    split=boosting.train_test_data_setup;fit=boosting.FedotLightGBMClassificationImplementation.fit
    def observed_split(value,*args,**kwargs):
        train,valid=split(value,*args,**kwargs)
        state['splits'].append(dict(train=np.asarray(train.idx).tolist(),evaluation=np.asarray(valid.idx).tolist()))
        return train,valid
    def observed_fit(implementation,input_data):
        state['splits']=[]
        before=time.monotonic();model=fit(implementation,input_data)
        dump=model.booster_.dump_model()
        record=dict(fold=state['fold'],seconds=time.monotonic()-before,
            input_rows=np.asarray(input_data.idx).tolist(),splits=state['splits'],
            params=model.get_params(),best_iteration=int(model.best_iteration_),
            leaves=[int(t['num_leaves']) for t in dump['tree_info']],dump=dump)
        fits.append(record)
        save(output/f'fit-{state["fold"]}.json',record)
        return model
    class Evaluator(PipelineObjectiveEvaluate):
        def prepare_graph(self,graph,train,fold_id=None,n_jobs=-1,**kwargs):
            state['fold']=fold_id
            return super().prepare_graph(graph,train,fold_id,n_jobs,**kwargs)
    def metric(graph,reference_data,fold_id=None,**kwargs):
        probs=np.asarray(graph.predict(reference_data).predict)
        truth=np.asarray(reference_data.target).ravel()
        arrays[f'probs_{fold_id}']=probs;arrays[f'truth_{fold_id}']=truth
        return probability_loss(probs,truth,arrays['classes'])
    evaluator=Evaluator(MetricsObjective(metric),splitter.build(data),time_constraint=timedelta(seconds=21.6),
        evaluation_time_constraint=timedelta(seconds=60),eval_n_jobs=8)
    random.seed(104735);np.random.seed(104735);before=time.monotonic()
    with patch.object(boosting,'train_test_data_setup',observed_split), \
            patch.object(boosting.FedotLightGBMClassificationImplementation,'fit',observed_fit):
        fitness=evaluator.evaluate(pipe)
    seconds=time.monotonic()-before
    with (output/'candidate.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    save(output/'result.json',dict(valid=bool(fitness.valid),fitness=float(fitness.value) if fitness.valid else None,
        seconds=seconds,smoke=smoke,graph=graph_record(pipe),expected_graph=candidate,
        fit_count=len(fits),input_rows=len(y),evaluation_cap=60,per_fit_cap=21.6,
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']}))


def audit_candidate(output,smoke=False):
    from sklearn.model_selection import StratifiedKFold
    r=json.loads((output/'result.json').read_text())
    assert r['smoke']==smoke and r['valid'] and r['fit_count']==3 and r['seconds']<=60
    assert r['graph']==r['expected_graph'] and r['graph'][0]['params']['min_data_in_leaf']==20
    losses=[];active=[];inner_counts=[]
    with np.load(output/'candidate.npz',allow_pickle=False) as z:
        y=z['target'];classes=np.unique(y)
        splits=StratifiedKFold(3,shuffle=True,random_state=42).split(y,y)
        for fold,(train,valid) in enumerate(splits):
            assert np.array_equal(z[f'train_rows_{fold}'],train) and np.array_equal(z[f'rows_{fold}'],valid)
            assert np.array_equal(z[f'truth_{fold}'],y[valid]) and np.array_equal(z['classes'],classes)
            losses.append(probability_loss(z[f'probs_{fold}'],y[valid],classes))
            f=json.loads((output/f'fit-{fold}.json').read_text())
            assert f['fold']==fold and f['input_rows']==train.tolist() and f['seconds']<=21.6
            assert f['params']['min_data_in_leaf']==20 and f['params']['n_jobs']==8
            assert len(f['splits'])==2 and f['splits'][0]==f['splits'][1]
            inner=f['splits'][0];assert not set(inner['train'])&set(inner['evaluation'])
            assert sorted(inner['train']+inner['evaluation'])==train.tolist()
            assert len(inner['train'])==int(.8*len(train))
            leaves=[int(t['num_leaves']) for t in f['dump']['tree_info']]
            assert leaves==f['leaves'] and leaves
            active.append(any(v>1 for v in leaves));inner_counts.append(len(inner['train']))
    assert abs(np.mean(losses)-r['fitness'])<1e-12
    return dict(fold_losses=losses,active_splits=active,inner_train_rows=inner_counts,mean=float(np.mean(losses)))


def audit(stage):
    count=verify(stage);output=stage/'candidate';p=json.loads((stage/'process.json').read_text())
    assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=90
    r=json.loads((output/'result.json').read_text());control=json.loads((stage/'control/evaluation.json').read_text())
    assert r['expected_graph']==candidate_graph(control['graph'])
    assert r['environment']==dict(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',FEDOT_OPERATION_DEFAULTS_OVERLAY=None)
    candidate=audit_candidate(output);losses={}
    with np.load(output/'candidate.npz',allow_pickle=False) as a:
        for path in sorted((stage/'control').glob('prediction-*.npz')):
            with np.load(path,allow_pickle=False) as z:
                fold=int(z['fold']);assert fold not in losses
                for k in ['rows','train_rows','truth']:assert np.array_equal(z[k],a[f'{k}_{fold}'])
                assert np.array_equal(z['classes'],a['classes'])
                assert np.allclose(z['probs'],1/3,rtol=0,atol=1e-12)
                losses[fold]=probability_loss(z['probs'],z['truth'],z['classes'])
    assert sorted(losses)==[0,1,2] and abs(np.mean(list(losses.values()))-control['fitness'])<1e-12
    assert candidate['inner_train_rows']==[48]*3
    control_losses=[losses[i] for i in range(3)]
    previous=json.loads((stage/'h29-audit.json').read_text())
    native=previous['sides']['prepared-false']['observation']['evaluations'][0]['fitness']
    return dict(valid=True,verified_files=count,model_fits=3,control_retrained=False,control_mean=control['fitness'],
        control_fold_losses=control_losses,candidate=candidate,h29_native_context_only=native,
        mechanism_passed=gate(control_losses,candidate['fold_losses'],candidate['active_splits']),
        process_seconds=p['seconds'],scope='Single 29->20 min_data_in_leaf intervention on explored synthetic H29 TRAIN; no overall native quality gain requirement, default/searchspace promotion or fabert transfer')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    before=time.monotonic();output=stage/'candidate';output.mkdir()
    env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
    env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
    with (stage/'worker.log').open('x') as log:
        start=time.monotonic();p=subprocess.Popen([sys.executable,str(stage/'leaf_capacity_probe.py'),'worker','--stage',str(stage)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:code=p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=p.returncode
    save(stage/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-start))
    assert code==0 and not timed_out
    report=audit(stage);report['stage_seconds']=time.monotonic()-before;assert report['stage_seconds']<=120
    save(stage/'independent-audit.json',report);save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit']);p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
    if a.mode=='worker':worker(a.stage,a.stage/'candidate')
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as error:save(a.stage/'FAILED',dict(error=repr(error)));raise
