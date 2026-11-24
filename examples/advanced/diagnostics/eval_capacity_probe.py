"""H31: one eval-set switch, retaining H29 leaf size and all model ceilings."""
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
from unittest.mock import patch

import numpy as np
from examples.advanced.diagnostics import leaf_capacity_probe as previous
from examples.advanced.diagnostics.cache_startup_contract import save, verify
from examples.advanced.diagnostics.cache_cost_probe import probability_loss


def candidate_graph(control):
    result=deepcopy(control)
    assert len(result)==2 and result[0]['operation']=='lgbm' and result[1]['operation']=='scaling'
    assert result[0]['params']['min_data_in_leaf']==29 and result[0]['params']['use_eval_set'] is True
    result[0]['params']['use_eval_set']=False
    return result


def worker(stage,output,smoke=False):
    from lightgbm import LGBMClassifier
    original=LGBMClassifier.fit
    fits=[]
    def observe(model,*args,**kwargs):
        x=kwargs.get('X',args[0] if args else None)
        y=kwargs.get('y',args[1] if len(args)>1 else None)
        record=dict(rows=len(x),columns=list(x.columns),target=np.asarray(y).ravel().tolist(),
            eval_set_absent=kwargs.get('eval_set') is None,
            callbacks=[type(c).__name__ for c in kwargs.get('callbacks',[])],params=model.get_params())
        result=original(model,*args,**kwargs)
        record.update(best_iteration=int(model.best_iteration_),trees=int(model.booster_.num_trees()),
                      iterations=int(model.booster_.current_iteration()))
        save(output/f'actual-fit-{len(fits)}.json',record);fits.append(record)
        return result
    with patch.object(previous,'candidate_graph',candidate_graph),patch.object(LGBMClassifier,'fit',observe):
        previous.worker(stage,output,smoke=smoke)
    assert len(fits)==3


def audit_candidate(output,smoke=False):
    from sklearn.model_selection import StratifiedKFold
    r=json.loads((output/'result.json').read_text())
    assert r['smoke']==smoke and r['valid'] and r['fit_count']==3 and r['seconds']<=60
    assert r['graph']==r['expected_graph']
    params=r['graph'][0]['params'];assert params['min_data_in_leaf']==29 and params['use_eval_set'] is False
    losses=[];active=[];actual_rows=[];iterations=[]
    with np.load(output/'candidate.npz',allow_pickle=False) as z:
        y=z['target'];classes=np.unique(y)
        for fold,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(y,y)):
            assert np.array_equal(z[f'train_rows_{fold}'],train) and np.array_equal(z[f'rows_{fold}'],valid)
            assert np.array_equal(z[f'truth_{fold}'],y[valid]) and np.array_equal(z['classes'],classes)
            losses.append(probability_loss(z[f'probs_{fold}'],y[valid],classes))
            f=json.loads((output/f'fit-{fold}.json').read_text())
            a=json.loads((output/f'actual-fit-{fold}.json').read_text())
            assert f['fold']==fold and f['input_rows']==train.tolist() and f['seconds']<=21.6
            assert f['splits']==[] and a['rows']==len(train) and a['eval_set_absent']
            assert a['target']==y[train].tolist() and a['params']==f['params']
            assert a['params']['min_data_in_leaf']==29 and a['params']['n_jobs']==8
            assert a['params']['early_stopping_rounds'] is None
            assert a['callbacks']==['_LightGBMFitTimeLimitCallback'] and a['best_iteration']==0
            # Every graph model parameter survives, except the existing no-eval normalization.
            for key,value in params.items():
                if key in ['enable_categorical','use_eval_set','early_stopping_rounds']:continue
                assert a['params'][key]==value,key
            leaves=[int(t['num_leaves']) for t in f['dump']['tree_info']]
            assert leaves==f['leaves'] and leaves and len(leaves)==a['trees']
            assert a['iterations']<=a['params']['n_estimators']
            active.append(any(v>1 for v in leaves));actual_rows.append(a['rows']);iterations.append(a['iterations'])
    assert abs(np.mean(losses)-r['fitness'])<1e-12
    return dict(fold_losses=losses,active_splits=active,actual_train_rows=actual_rows,
                iterations=iterations,mean=float(np.mean(losses)))


def audit(stage):
    count=verify(stage);out=stage/'candidate';p=json.loads((stage/'process.json').read_text())
    assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=90
    r=json.loads((out/'result.json').read_text());control=json.loads((stage/'control/evaluation.json').read_text())
    assert r['expected_graph']==candidate_graph(control['graph'])
    assert r['environment']==dict(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',FEDOT_OPERATION_DEFAULTS_OVERLAY=None)
    candidate=audit_candidate(out);losses={}
    with np.load(out/'candidate.npz',allow_pickle=False) as a:
        for path in sorted((stage/'control').glob('prediction-*.npz')):
            with np.load(path,allow_pickle=False) as z:
                fold=int(z['fold']);assert fold not in losses
                for k in ['rows','train_rows','truth']:assert np.array_equal(z[k],a[f'{k}_{fold}'])
                assert np.array_equal(z['classes'],a['classes']) and np.allclose(z['probs'],1/3,rtol=0,atol=1e-12)
                losses[fold]=probability_loss(z['probs'],z['truth'],z['classes'])
    assert sorted(losses)==[0,1,2] and abs(np.mean(list(losses.values()))-control['fitness'])<1e-12
    assert candidate['actual_train_rows']==[60]*3
    h30=json.loads((stage/'h30-audit.json').read_text());assert h30['mechanism_passed']
    control_losses=[losses[i] for i in range(3)]
    return dict(valid=True,verified_files=count,model_fits=3,control_retrained=False,
        control_mean=control['fitness'],control_fold_losses=control_losses,candidate=candidate,
        h30_context_only=h30['candidate']['mean'],h29_native_context_only=h30['h29_native_context_only'],
        mechanism_passed=previous.gate(control_losses,candidate['fold_losses'],candidate['active_splits']),
        process_seconds=p['seconds'],scope='Single eval-set switch at fixed leaf29; explored synthetic TRAIN. Existing early-stopping removal is inseparable; no pure row-count causality, quality search or default promotion.')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    before=time.monotonic();(stage/'candidate').mkdir()
    env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
    env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
    with (stage/'worker.log').open('x') as log:
        start=time.monotonic();p=subprocess.Popen([sys.executable,str(stage/'eval_capacity_probe.py'),'worker','--stage',str(stage)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
