"""Read-only H21 audit, run only after both full jobs terminate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def read(path):
    return json.loads(path.read_text())


def audit_observation(directory):
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.integrated_refit_probe import check_final_constructor
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    fit=read(directory/'fit.json')
    assert fit['error'] is None and fit['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
    assert fit['seconds']<=fit['api_seconds']
    if not (directory/'tuning-entry.json').exists():
        return dict(tuning_entered=False,policy_enabled=fit['policy_enabled'],mechanism_passed=False)
    entry=read(directory/'tuning-entry.json');exit=read(directory/'tuning-exit.json')
    final=read(directory/'final-fit.json');returned=read(directory/'tuning-return.json')['graph']
    assert final['graph']==returned==fit['final_graph']
    assert final['seconds']<=entry['per_fit_cap']
    history=OptHistory.load(str(directory/'history.json'));adapter=PipelineAdapter()
    assert graph_record(adapter.restore(history.generations[-1][0].graph))==returned
    choices=[g for g in history.generations if g.label=='final_choices']
    if choices:
        assert len(choices)==1 and graph_record(adapter.restore(choices[0][0].graph))==entry['graph']
    valid_proposals=sum(ind.fitness.valid for g in history.generations
                        if str(g.label).startswith('tuning_iteration_') for ind in g)
    with np.load(directory/'train-target.npz',allow_pickle=False) as z:
        y=z['target'].copy();rows=z['rows'].copy()
    assert np.array_equal(rows,np.arange(len(y))) and entry['rows']==len(y)
    classes=np.unique(y);splits=list(StratifiedKFold(entry['cv_folds'],shuffle=True,random_state=42).split(y,y))
    audited=[];all_probs=[]
    for i,evaluation in enumerate(exit['evaluations']):
        cell=directory/'cv'/f'{i:03d}'
        assert read(cell/'evaluation.json')==evaluation
        losses={};probs={}
        for path in sorted(cell.glob('prediction-*.npz')):
            with np.load(path,allow_pickle=False) as z:
                fold=int(z['fold']);train,valid=splits[fold]
                assert fold not in losses and np.array_equal(z['classes'],classes)
                assert np.array_equal(z['rows'],valid) and np.array_equal(z['train_rows'],train)
                assert np.array_equal(z['truth'],y[valid])
                p=z['probs'];assert p.shape==(len(valid),len(classes))
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                assert np.max(np.abs(p.sum(axis=1)-1))<=1e-6
                losses[fold]=float(log_loss(y[valid],p,labels=classes));probs[fold]=p.copy()
        if evaluation.get('valid'):
            assert len(losses)==len(splits)
            assert abs(np.mean(list(losses.values()))-evaluation['fitness'])<1e-10
        audited.append(dict(valid=evaluation.get('valid',False),fold_losses=losses,
                            partial_folds=len(losses),fitness=evaluation.get('fitness')))
        all_probs.append(probs)
    policy=exit['policy'];selected=False;fold_guard=False;proper_selection=False
    if policy and policy.get('status')=='evaluated' and len(policy['calls'])==3 and all(c['valid'] for c in policy['calls']):
        assert len(audited)==3 and all(c['valid'] for c in audited)
        calls=policy['calls'];native=entry['graph'];proposal=deepcopy(native)
        proposal[0]['params']['use_eval_set']=False
        assert [c['graph'] for c in calls]==[native,proposal,proposal]
        assert [e['graph'] for e in exit['evaluations']]==[native,proposal,proposal]
        assert all(np.array_equal(all_probs[1][f],all_probs[2][f]) for f in all_probs[1])
        selected=bool(calls[2]['fitness']<=.99*calls[0]['fitness'])
        assert returned==(proposal if selected else native)
        proper_selection=True
        fold_guard=all(a<=1.02*b for a,b in zip(audited[2]['fold_losses'].values(),audited[0]['fold_losses'].values()))
        bounds=policy['bounds'];remaining=fit['api_seconds']-entry['elapsed']
        assert abs(bounds['remaining']-remaining)<1.0
        assert abs(bounds['allocation']-min(1100,bounds['remaining']-bounds['refit_reserve']-bounds['postfit_reserve']))<1e-8
        assert bounds['refit_reserve']==entry['per_fit_cap']
        assert bounds['initial_cap']==min(300,(bounds['allocation']-100)/3)
        cap=min(bounds['initial_cap'],max(5,1.3*calls[0]['seconds']))
        assert calls[1]['cap']==calls[2]['cap']==cap
        assert policy['admission']['remaining']>=2*(cap+50)
        assert policy['charged_tuning_seconds']<=bounds['allocation']
        assert policy['evaluations_count']==1 and valid_proposals==1
        for number in [1,2]:
            cell=directory/'cv'/f'{number:03d}'
            for fold,(train,_) in enumerate(splits):
                paths=list((cell/f'pool-{fold}').glob('*.json'));assert len(paths)==1
                pool=read(paths[0]);assert pool['trees']==native[0]['params']['num_trees']
                assert pool['callback_count']==1 and len(pool['pools'])==1
                assert pool['pools'][0]['rows']==train.tolist()
                check_final_constructor(pool['params'],proposal[0]['params'],True)
        if selected:
            paths=list((directory/'final-pools').glob('*.json'));assert len(paths)==1
            pool=read(paths[0]);assert pool['trees']==native[0]['params']['num_trees']
            assert pool['callback_count']==1 and len(pool['pools'])==1
            assert pool['pools'][0]['rows']==rows.tolist()
            check_final_constructor(pool['params'],proposal[0]['params'],True)
    return dict(tuning_entered=True,policy_enabled=fit['policy_enabled'],entry_graph=entry['graph'],
        final_graph=returned,valid_proposals=int(valid_proposals),evaluations=audited,
        selected_allrow=selected,proper_selection=proper_selection,fold_guard=fold_guard,
        mechanism_passed=bool(selected and proper_selection and fold_guard),
        api_fit_seconds=fit['seconds'],final_refit_seconds=final['seconds'])


def audit(stage):
    from examples.advanced.diagnostics.run_paid_policy_pair import verify
    from examples.advanced.diagnostics.audit_followup import audit_run
    assert (stage/'QUEUE_FINISHED').exists()
    verify(stage)
    observations={side:audit_observation(stage/'runs'/side/'observation') for side in ['baseline','candidate']}
    assert observations['baseline']['policy_enabled'] is False
    assert observations['candidate']['policy_enabled'] is True
    for side in ['baseline','candidate']:
        process=read(stage/'runs'/side/'process.json')
        assert process['exit_code']==0 and not process['timed_out'] and process['seconds']<=3900
    outer={};truth={}
    for side in ['baseline','candidate']:
        outer[side],truth[side]=audit_run(stage/'runs'/side)
    assert np.array_equal(truth['baseline'],truth['candidate'])
    assert outer['baseline']['csv_columns']==outer['candidate']['csv_columns']
    outer.update(paired_truth=True,paired_column_order=True)
    matched=observations['baseline'].get('entry_graph')==observations['candidate'].get('entry_graph')
    native_match=observations['candidate'].get('entry_graph')==read(stage/'h19-result.json')['policy']['initial_graph']
    if observations['candidate']['mechanism_passed'] and native_match:
        directory=stage/'runs/candidate/observation'
        with np.load(directory/'train-target.npz',allow_pickle=False) as z:
            assert len(z['target'])==7413 and len(np.unique(z['target']))==7
        paths=list((directory/'final-pools').glob('*.json'))
        paths+=list((directory/'cv/001').glob('pool-*/*.json'))
        paths+=list((directory/'cv/002').glob('pool-*/*.json'))
        assert len(paths)==7
        for path in paths:
            p=read(path)
            assert p['trees']==3000 and p['pools'][0]['categories']==list(range(715,800))
    extra=observations['candidate'].get('valid_proposals',0)-observations['baseline'].get('valid_proposals',0)
    return dict(observations=observations,outer=outer,matched_composition_graph=matched,
        additional_valid_proposals=extra,
        native_graph_match=native_match,
        mechanism_gate=bool(matched and native_match and extra>=1 and observations['candidate']['mechanism_passed']),
        outer_descriptive_gate=outer['candidate']['logloss']<=.99*outer['baseline']['logloss'],
        scope='One prospective full policy pair; fold7 labels overlap earlier explored TRAIN; no independent-data or speed claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit(args.stage)))
