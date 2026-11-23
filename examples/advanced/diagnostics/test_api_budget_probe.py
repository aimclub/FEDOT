import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from examples.advanced.diagnostics import api_budget_probe as probe
from examples.advanced.diagnostics.cache_cost_probe import sha
from sklearn.model_selection import StratifiedKFold


def test_cap_and_reservation():
    assert probe.calibrated_cap(200)==260 and probe.calibrated_cap(290)==300
    for bad in [0,-1,float('inf'),float('nan')]:
        with pytest.raises(ValueError):probe.calibrated_cap(bad)
    assert probe.admitted(700,300) and not probe.admitted(699,300)


def test_selection_needs_valid_measured_incumbent():
    valid=lambda score:dict(valid=True,evaluation=dict(fitness=score))
    assert probe.select(valid(1),valid(.9))==probe.PROPOSAL
    assert probe.select(valid(1),valid(1.1))=={}
    assert probe.select(dict(valid=False),valid(.9))=={}


@pytest.mark.parametrize('policy,initial_valid',[('baseline',False),('candidate',True),('candidate',False)])
def test_policy_fixed_order_and_calibration(tmp_path,monkeypatch,policy,initial_valid):
    calls=[]
    def fake(stage,p,label,cap,params,deadline):
        calls.append((label,cap,params))
        return dict(valid=initial_valid if label=='initial' else True,
                    evaluation=dict(seconds=200,fitness=1. if label=='initial' else .95))
    monkeypatch.setattr(probe,'execute',fake)
    r=probe.policy(tmp_path,policy)
    if policy=='candidate' and not initial_valid:
        assert len(calls)==1 and r['status']=='calibration_failed'
    else:
        assert [c[0] for c in calls]==['initial','proposal','confirmation']
        assert calls[1][1]==(152 if policy=='baseline' else 260)
        assert calls[1][2]==probe.PROPOSAL
        assert calls[2][2]==(probe.PROPOSAL if initial_valid else {})


def audit_fixture(stage,fault=None):
    (stage/'cells').mkdir();np.savez(stage/'train.npz',target=np.arange(7413)%7)
    for name in ['baseline','candidate']:
        c=dict(name='fresh_'+name+'_initial',valid=False,exit_code=0,status='exited',external_cap=190 if name=='baseline' else 340,
               evaluation_cap=152. if name=='baseline' else 300.,process_seconds=180 if name=='baseline' else 310)
        if name=='candidate':
            r=dict(status='calibration_failed',initial=c,valid_proposals=0,policy_seconds=311)
        else:
            proposal=dict(valid=False,status='not_admitted')
            final={**c,'name':'fresh_baseline_confirmation'}
            r=dict(status='evaluated',initial=c,proposal=proposal,confirmation=final,selected_params={},
                   next_cap=152.,proposal_remaining=100,valid_proposals=0,policy_seconds=361)
        if fault=='charge':r['policy_seconds']=1
        if fault=='count':r['valid_proposals']=1
        if fault=='cap' and name=='baseline':r['next_cap']=153
        if fault=='selection' and name=='baseline':r['selected_params']=probe.PROPOSAL
        if fault=='admission' and name=='baseline':r['proposal_remaining']=1000
        for label in ['initial','confirmation']:
            if label not in r:continue
            v=r[label];p=stage/'cells'/v['name']
            p.with_suffix('.process.json').write_text(json.dumps(v))
            p.with_suffix('.spec.json').write_text(json.dumps(dict(cap=v['evaluation_cap'],params={},split_seed=probe.SPLIT_SEED)))
        (stage/(name+'.json')).write_text(json.dumps(r))


def test_failed_calibration_is_no_quality_result(tmp_path):
    audit_fixture(tmp_path)
    assert not probe.audit(tmp_path)['gate_passed']


@pytest.mark.parametrize('fault',['charge','count','cap','selection','admission'])
def test_budget_auditor_rejects_corruption(tmp_path,fault):
    audit_fixture(tmp_path,fault)
    with pytest.raises(AssertionError):probe.audit(tmp_path)


@pytest.mark.parametrize('fault',[None,'probs','rows','params','score'])
def test_independent_native_npz_audit(tmp_path,fault):
    (tmp_path/'cells').mkdir();y=np.arange(7413)%7
    np.savez(tmp_path/'train.npz',target=y)
    graph="(/n_scaling;)/n_catboost_{'num_trees': 3000, 'use_eval_set': True}"
    (tmp_path/'native-params-reference.json').write_text(json.dumps(dict(graph_id_before=graph)))
    arrays={'classes':np.arange(7)}
    for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=probe.SPLIT_SEED).split(y,y)):
        p=np.full((len(valid),7),.2/6);p[np.arange(len(valid)),y[valid]]=.8
        if fault=='probs':p[0]=0
        arrays.update({f'rows_{i}':valid[::-1] if fault=='rows' else valid,f'train_rows_{i}':train,
                       f'truth_{i}':y[valid],f'probs_{i}':p})
    name='fresh_candidate_initial';path=tmp_path/'cells'/(name+'.json')
    np.savez(path.with_suffix('.npz'),**arrays)
    r=dict(valid=True,folds=[{'fitted_nodes':[dict(operation='catboost',cat_feature_indices=list(range(715,800)),tree_count_=2999)]}]*3,
           split_seed=probe.SPLIT_SEED,seconds=2,evaluation_cap=300,cache_enabled=False,cache_before={},cache_after={},node_cache_before={},node_cache_after={},
           graph_id_before=graph,graph_id_after=graph,parameter_override={},data_sha256=sha(tmp_path/'train.npz'),npz_sha256=sha(path.with_suffix('.npz')),fitness=float(-np.log(.8)))
    if fault=='params':r['graph_id_after']='changed'
    if fault=='score':r['fitness']+=.1
    path.write_text(json.dumps(r));record=dict(evaluation=r,process_seconds=3)
    if fault:
        with pytest.raises((AssertionError,ValueError)):probe.audit_valid_cell(tmp_path,name,record,{},300,y)
    else:assert abs(probe.audit_valid_cell(tmp_path,name,record,{},300,y)['cv']+np.log(.8))<1e-12


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'),reason='host-only real child transport')
def test_actual_api_child_accounting_and_allrow_cv(tmp_path):
    import pandas as pd
    harness=Path(os.environ['FEDOT_H9_ADAPTER']).parents[2]
    (tmp_path/'harness').symlink_to(harness,target_is_directory=True)
    x=np.random.RandomState(104741).normal(size=(60,4)).astype('float32');x[:,2:]=np.arange(120).reshape(60,2)%3
    y=np.arange(60)%3
    np.savez(tmp_path/'train.npz',features=x,target=y)
    pd.DataFrame(x).to_pickle(tmp_path/'train-frame.pkl');(tmp_path/'cells').mkdir()
    r=probe.execute(tmp_path,'candidate','proposal',30,{**probe.PROPOSAL,'enable_categorical':True},time.monotonic()+100,smoke=True)
    assert r['valid'] and r['exit_code']==0 and r['process_seconds']>r['evaluation']['seconds']
    assert r['evaluation']['split_seed']==probe.SPLIT_SEED and len(r['evaluation']['folds'])==3
    pools=list((tmp_path/'pool-audits/fresh_candidate_proposal/pools').glob('*.json'));assert len(pools)==3
    for p in pools:
        v=json.loads(p.read_text());assert v['trees']==8 and len(v['pools'])==1
        assert v['pools'][0]['count']==40 and v['pools'][0]['categories']==[2,3]
