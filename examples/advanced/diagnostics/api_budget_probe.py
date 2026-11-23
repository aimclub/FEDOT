"""H15: paid fresh-cost calibration on actual API data and new CV splits."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
from examples.advanced.diagnostics.allrow_fit_probe import instrument_fit
from examples.advanced.diagnostics.cache_cost_probe import worker, sha, write_json, probability_loss
from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
from examples.advanced.diagnostics.subspace_validation_probe import graph_params, quality_gate

SPLIT_SEED = 104741
POLICY_SECONDS = 1100
BASELINE_CAP = 152.
CALIBRATION_CAP = 300.
PROPOSAL = {'use_eval_set': False}


def calibrated_cap(seconds):
    if not math.isfinite(seconds) or seconds <= 0:raise ValueError('Invalid calibration')
    return min(CALIBRATION_CAP, 1.3*seconds)


def admitted(remaining, cap):
    return remaining >= 2*(cap+50)


def select(initial, proposal):
    if initial.get('valid') and proposal.get('valid') and proposal['evaluation']['fitness'] < initial['evaluation']['fitness']:
        return PROPOSAL.copy()
    return {}


def cell(stage, name, smoke=False):
    import pandas as pd
    spec=json.loads((stage/'cells'/(name+'.spec.json')).read_text())
    with np.load(stage/'train.npz',allow_pickle=False) as z:y=z['target'].copy()
    data=capture(load_adapter(stage),pd.read_pickle(stage/'train-frame.pkl'),y)
    write_json(stage/'cells'/(name+'.input.json'),dict(schema=summarize(data['data']),params=data['params']))
    poolstage=stage/'pool-audits'/name
    (poolstage/'pools').mkdir(parents=True)
    instrument_fit(poolstage)
    worker(stage,name,smoke=smoke,spec=spec,input_data=data['data'])


def execute(stage, policy, label, cap, params, deadline, smoke=False):
    name='fresh_'+policy+'_'+label
    spec=dict(cap=cap,params=params,split_seed=SPLIT_SEED)
    write_json(stage/'cells'/(name+'.spec.json'),spec)
    external=min(cap+45,max(0,deadline-time.monotonic()-5))
    if external<=1:raise RuntimeError('No confirmation budget remains')
    before=time.monotonic()
    with (stage/'cells'/(name+'.log')).open('x') as log:
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--stage',str(stage),'--name',name]
        if smoke:command.append('--smoke')
        p=subprocess.Popen(command,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status='exited'
        try:code=p.wait(timeout=external)
        except subprocess.TimeoutExpired:
            status='external_timeout';os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=p.returncode
    record=dict(name=name,status=status,exit_code=code,external_cap=external,evaluation_cap=cap,
                process_seconds=time.monotonic()-before,valid=False)
    path=stage/'cells'/(name+'.json')
    if code==0 and path.exists():
        evaluation=json.loads(path.read_text());record.update(valid=evaluation['valid'],evaluation=evaluation)
    write_json(stage/'cells'/(name+'.process.json'),record)
    return record


def policy(stage, name):
    before=time.monotonic();deadline=before+POLICY_SECONDS
    initial=execute(stage,name,'initial',BASELINE_CAP if name=='baseline' else CALIBRATION_CAP,{},deadline)
    result=dict(initial=initial,valid_proposals=0)
    if name=='candidate' and not initial['valid']:
        result['status']='calibration_failed'
    else:
        cap=BASELINE_CAP if name=='baseline' else calibrated_cap(initial['evaluation']['seconds'])
        remaining=deadline-time.monotonic()
        proposal=execute(stage,name,'proposal',cap,PROPOSAL,deadline) if admitted(remaining,cap) else dict(valid=False,status='not_admitted')
        selected=select(initial,proposal)
        final=execute(stage,name,'confirmation',cap,selected,deadline)
        result.update(status='evaluated',proposal=proposal,confirmation=final,selected_params=selected,
                      next_cap=cap,proposal_remaining=remaining,valid_proposals=int(proposal['valid']))
    result['policy_seconds']=time.monotonic()-before
    write_json(stage/(name+'.json'),result)
    return result


def audit_valid_cell(stage, name, record, params, cap, truth):
    path=stage/'cells'/(name+'.json')
    r=json.loads(path.read_text());assert r==record['evaluation']
    assert r['valid'] and len(r['folds'])==3 and r['split_seed']==SPLIT_SEED
    assert 0<r['seconds']<=record['process_seconds'] and r['evaluation_cap']==cap
    assert not any(r[k] for k in ['cache_enabled','cache_before','cache_after','node_cache_before','node_cache_after'])
    native=graph_params(json.loads((stage/'native-params-reference.json').read_text())['graph_id_before'])
    assert r['graph_id_before']==r['graph_id_after'] and graph_params(r['graph_id_before'])=={**native,**params}
    assert r['parameter_override']==params and r['data_sha256']==sha(stage/'train.npz')
    assert r['npz_sha256']==sha(path.with_suffix('.npz'))
    losses=[];trees=[]
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        assert np.array_equal(z['classes'],np.unique(truth))
        for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(truth,truth)):
            assert np.array_equal(z[f'rows_{i}'],valid) and np.array_equal(z[f'train_rows_{i}'],train)
            assert np.array_equal(z[f'truth_{i}'],truth[valid])
            losses.append(probability_loss(z[f'probs_{i}'],truth[valid],z['classes']))
            fitted=[n for n in r['folds'][i]['fitted_nodes'] if n['operation']=='catboost']
            assert len(fitted)==1 and fitted[0]['cat_feature_indices']==list(range(715,800))
            tree=fitted[0]['tree_count_'];assert (tree==3000 if params else 0<tree<=3000)
            trees.append(tree)
    assert abs(float(np.mean(losses))-r['fitness'])<1e-10
    if params:
        pooldir=stage/'pool-audits'/name/'pools';poolfiles=list(pooldir.glob('*.json'));assert len(poolfiles)==3
        import hashlib
        for train,_ in StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(truth,truth):
            key=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
            p=json.loads((pooldir/(key+'.json')).read_text());assert p['trees']==3000 and p['callback_count']==1
            assert len(p['pools'])==1
            pool=p['pools'][0]
            assert pool['rows']==train.tolist() and pool['count']==4942 and pool['categories']==list(range(715,800))
            expected={k:v for k,v in native.items() if k not in {'n_jobs','use_eval_set','enable_categorical','callbacks','fit_time_limit','early_stopping_rounds','od_wait','od_type','od_pval'}}
            expected.update(use_best_model=False,thread_count=8)
            assert all(p['model_params'].get(k)==v for k,v in expected.items())
            assert not set(p['model_params'])&{'early_stopping_rounds','od_wait','od_type','od_pval'}
    return dict(cv=float(np.mean(losses)),folds=losses,trees=trees,seconds=r['seconds'],prediction_sha256=r['npz_sha256'])


def audit(stage):
    with np.load(stage/'train.npz',allow_pickle=False) as z:truth=z['target'].ravel()
    assert len(truth)==7413
    policies={};valid={};invalid={}
    for name in ['baseline','candidate']:
        r=json.loads((stage/(name+'.json')).read_text());policies[name]=r
        assert 0<r['policy_seconds']<=POLICY_SECONDS
        assert r['valid_proposals']==int(r.get('proposal',{}).get('valid',False))
        assert sum(r.get(k,{}).get('process_seconds',0) for k in ['initial','proposal','confirmation'])<=r['policy_seconds']+.01
        if r['status']=='calibration_failed':assert name=='candidate' and not r['initial']['valid']
        else:
            assert r['status']=='evaluated'
            assert r['next_cap']==(BASELINE_CAP if name=='baseline' else calibrated_cap(r['initial']['evaluation']['seconds']))
            assert ('name' in r['proposal'])==admitted(r['proposal_remaining'],r['next_cap'])
            assert r['selected_params']==select(r['initial'],r['proposal'])
        for label in ['initial','proposal','confirmation']:
            c=r.get(label,{})
            if 'name' not in c:continue
            cellname='fresh_'+name+'_'+label;assert c['name']==cellname
            cap=(BASELINE_CAP if name=='baseline' else CALIBRATION_CAP) if label=='initial' else r['next_cap']
            params={} if label=='initial' else PROPOSAL if label=='proposal' else r['selected_params']
            assert c==json.loads((stage/'cells'/(cellname+'.process.json')).read_text())
            assert json.loads((stage/'cells'/(cellname+'.spec.json')).read_text())==dict(cap=cap,params=params,split_seed=SPLIT_SEED)
            assert c['evaluation_cap']==cap and 0<c['external_cap']<=cap+45
            assert 0<c['process_seconds']<=c['external_cap']+6
            evaluation_path=stage/'cells'/(cellname+'.json')
            if c['exit_code']==0 and evaluation_path.exists():
                actual=json.loads(evaluation_path.read_text())
                assert c.get('evaluation')==actual and c['valid']==actual['valid']
            if c['valid']:
                assert c['exit_code']==0
                valid[cellname]=audit_valid_cell(stage,cellname,c,params,cap,truth)
            else:invalid[cellname]=dict(status=c['status'],folds=len(c.get('evaluation',{}).get('folds',[])))
    b=policies['candidate'];initial=valid.get('fresh_candidate_initial');final=valid.get('fresh_candidate_confirmation')
    quality=bool(initial and final and b.get('selected_params')==PROPOSAL and quality_gate(initial['cv'],final['cv'],initial['folds'],final['folds']))
    if final and b.get('selected_params')==PROPOSAL:
        a=stage/'cells/fresh_candidate_proposal.npz';c=stage/'cells/fresh_candidate_confirmation.npz'
        with np.load(a) as x,np.load(c) as y:assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
    return dict(valid_cells=valid,invalid_cells=invalid,
                policy_seconds={k:r['policy_seconds'] for k,r in policies.items()},
                valid_proposals={k:r['valid_proposals'] for k,r in policies.items()},
                selected_params={k:r.get('selected_params') for k,r in policies.items()},
                next_caps={k:r.get('next_cap') for k,r in policies.items()},
                quality_replication_passed=quality,
                gate_passed=quality and b['valid_proposals']>policies['baseline']['valid_proposals'],
                scope='Actual API paid budget/new split on explored TRAIN; no outer/untouched/speed conclusion')


def suite(stage):
    import fcntl
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=[a.decode(errors='replace') for a in p.read_bytes().split(b'\0') if a]
            except OSError:continue
            pid=int(p.parent.name);parent=pid==os.getppid() and args and Path(args[0]).name=='timeout' and own in args
            extra=any(Path(a).name in ['api_budget_probe.py','allrow_fit_probe.py','noeval_probe.py','catboost_ceiling_probe.py','adapter_cv_probe.py','subspace_validation_probe.py'] for a in args)
            if pid!=os.getpid() and not parent and (extra or training_conflict(pid,args,os.getpid(),os.getppid(),own)):raise RuntimeError('Host occupied: '+str(pid))
        m=json.loads((stage/'manifest.json').read_text());assert m['preflight_passed']
        for n,h in m['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        before=time.monotonic()
        try:
            for name in ['baseline','candidate']:policy(stage,name)
            a=audit(stage);a['stage_seconds']=time.monotonic()-before;assert a['stage_seconds']<=2250
            write_json(stage/'independent-audit.json',a)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as e:write_json(stage/'FAILED',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','worker','audit']);p.add_argument('--stage',type=Path,required=True);p.add_argument('--name');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.mode=='suite':suite(a.stage)
    elif a.mode=='worker':cell(a.stage,a.name,a.smoke)
    else:print(json.dumps(audit(a.stage)))
