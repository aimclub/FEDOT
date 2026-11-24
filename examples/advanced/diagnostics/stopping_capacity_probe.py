"""H32: isolate stopping from H31's coupled eval-set switch; no quality search."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from examples.advanced.diagnostics import leaf_capacity_probe as base
from examples.advanced.diagnostics.cache_startup_contract import save,verify
from examples.advanced.diagnostics.cache_cost_probe import probability_loss


def candidate_graph(control):
    g=deepcopy(control)
    assert len(g)==2 and g[0]['operation']=='lgbm' and g[1]['operation']=='scaling'
    p=g[0]['params'];assert p['min_data_in_leaf']==29 and p['use_eval_set'] is True
    assert isinstance(p['early_stopping_rounds'],int) and p['early_stopping_rounds']>0
    p['early_stopping_rounds']=None
    return g


def worker(stage,output,smoke=False):
    from lightgbm import LGBMClassifier
    original=LGBMClassifier.fit;count=[]
    def observe(model,*args,**kwargs):
        x=kwargs.get('X',args[0] if args else None);y=kwargs.get('y',args[1] if len(args)>1 else None)
        ev=kwargs['eval_set'];assert len(ev)==1
        r=dict(rows=len(x),target=np.asarray(y).ravel().tolist(),eval_rows=len(ev[0][0]),
            eval_target=np.asarray(ev[0][1]).ravel().tolist(),params=model.get_params(),
            callbacks=[type(c).__name__ for c in kwargs.get('callbacks',[])])
        fitted=original(model,*args,**kwargs)
        r.update(iterations=int(model.booster_.current_iteration()),trees=int(model.booster_.num_trees()),best_iteration=int(model.best_iteration_))
        save(output/f'actual-fit-{len(count)}.json',r);count.append(r);return fitted
    with patch.object(base,'candidate_graph',candidate_graph),patch.object(LGBMClassifier,'fit',observe):
        base.worker(stage,output,smoke=smoke)
    assert len(count)==3


def audit_candidate(output,smoke=False):
    from sklearn.model_selection import StratifiedKFold
    r=json.loads((output/'result.json').read_text());assert r['smoke']==smoke and r['valid'] and r['fit_count']==3 and r['seconds']<=60
    assert r['graph']==r['expected_graph'];params=r['graph'][0]['params']
    assert params['min_data_in_leaf']==29 and params['use_eval_set'] is True and params['early_stopping_rounds'] is None
    losses=[];active=[];uniform=[];rows=[];iterations=[]
    with np.load(output/'candidate.npz',allow_pickle=False) as z:
        y=z['target'];classes=np.unique(y)
        for fold,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(y,y)):
            assert np.array_equal(z[f'train_rows_{fold}'],train) and np.array_equal(z[f'rows_{fold}'],valid)
            assert np.array_equal(z[f'truth_{fold}'],y[valid]) and np.array_equal(z['classes'],classes)
            probs=z[f'probs_{fold}'];losses.append(probability_loss(probs,y[valid],classes))
            f=json.loads((output/f'fit-{fold}.json').read_text());a=json.loads((output/f'actual-fit-{fold}.json').read_text())
            assert f['fold']==fold and f['input_rows']==train.tolist() and f['seconds']<=21.6
            assert len(f['splits'])==2 and f['splits'][0]==f['splits'][1]
            split=f['splits'][0];assert not set(split['train'])&set(split['evaluation'])
            assert sorted(split['train']+split['evaluation'])==train.tolist()
            assert a['rows']==len(split['train'])==int(.8*len(train)) and a['eval_rows']==len(split['evaluation'])
            assert a['target']==y[split['train']].tolist() and a['eval_target']==y[split['evaluation']].tolist()
            assert a['params']==f['params'] and a['params']['n_jobs']==8
            for key,value in params.items():
                if key in ['use_eval_set','enable_categorical']:continue
                assert a['params'][key]==value,key
            assert a['callbacks']==['_LightGBMFitTimeLimitCallback'] and a['best_iteration']==0
            leaves=[int(t['num_leaves']) for t in f['dump']['tree_info']]
            assert leaves==f['leaves'] and leaves and len(leaves)==a['trees']
            assert a['iterations']<=a['params']['n_estimators']
            active.append(any(v>1 for v in leaves));uniform.append(bool(np.allclose(probs,1/3,rtol=0,atol=1e-12)))
            rows.append(a['rows']);iterations.append(a['iterations'])
    assert abs(np.mean(losses)-r['fitness'])<1e-12
    return dict(fold_losses=losses,active_splits=active,uniform=uniform,actual_train_rows=rows,iterations=iterations,mean=float(np.mean(losses)))


def gate(candidate,matched):
    return bool(len(candidate['uniform'])==3 and all(candidate['uniform']) and not any(candidate['active_splits']) and matched)


def audit(stage):
    count=verify(stage);out=stage/'candidate';p=json.loads((stage/'process.json').read_text())
    assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=90
    r=json.loads((out/'result.json').read_text());c=json.loads((stage/'control/evaluation.json').read_text())
    assert r['expected_graph']==candidate_graph(c['graph'])
    assert r['environment']==dict(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',FEDOT_OPERATION_DEFAULTS_OVERLAY=None)
    candidate=audit_candidate(out);losses={};matched=True
    with np.load(out/'candidate.npz',allow_pickle=False) as a:
        for path in sorted((stage/'control').glob('prediction-*.npz')):
            with np.load(path,allow_pickle=False) as z:
                fold=int(z['fold']);assert fold not in losses
                for key in ['rows','train_rows','truth']:assert np.array_equal(z[key],a[f'{key}_{fold}'])
                assert np.array_equal(z['classes'],a['classes'])
                losses[fold]=probability_loss(z['probs'],z['truth'],z['classes'])
                matched=matched and bool(np.allclose(z['probs'],a[f'probs_{fold}'],rtol=0,atol=1e-12))
    assert sorted(losses)==[0,1,2] and abs(np.mean(list(losses.values()))-c['fitness'])<1e-12
    assert candidate['actual_train_rows']==[48]*3
    assert json.loads((stage/'h31-audit.json').read_text())['mechanism_passed']
    return dict(valid=True,verified_files=count,model_fits=3,control_retrained=False,candidate=candidate,
        control_mean=c['fitness'],control_probability_match=matched,mechanism_passed=gate(candidate,matched),
        process_seconds=p['seconds'],scope='Early stopping alone insufficient on fixed H29 synthetic TRAIN: prospective negative-control mechanism; not universal causality or quality search.')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage);save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    before=time.monotonic();(stage/'candidate').mkdir();env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
    env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
    with (stage/'worker.log').open('x') as log:
        start=time.monotonic();p=subprocess.Popen([sys.executable,str(stage/'stopping_capacity_probe.py'),'worker','--stage',str(stage)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);timed_out=False
        try:code=p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=p.returncode
    save(stage/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-start));assert code==0 and not timed_out
    report=audit(stage);report['stage_seconds']=time.monotonic()-before;assert report['stage_seconds']<=120
    save(stage/'independent-audit.json',report);save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit']);p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
    if a.mode=='worker':worker(a.stage,a.stage/'candidate')
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as error:save(a.stage/'FAILED',dict(error=repr(error)));raise
