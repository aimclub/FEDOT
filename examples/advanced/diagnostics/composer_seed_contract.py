"""H26 real factory/composer seed routing to original and paid tuners, no fits."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
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

from examples.advanced.diagnostics.cache_startup_contract import save,verify
from examples.advanced.diagnostics.fallback_rng_contract import analytic_score,canonical_history,graph_from_record

CELLS=('original-none','original-seeded','paid-none','paid-seeded')


def run_case(stage,output,name,original_graph=None,smoke=False):
    from fedot.api.main import Fedot
    from fedot.api.time import ApiTime
    from fedot.api.api_utils import api_composer
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.tuning import simultaneous,hyperopt_tuner
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    assert name in CELLS
    paid=name.startswith('paid-');seed=104758 if name.endswith('-seeded') else None
    # Exercise both AMLB-style string and structured integer inputs. Null/absent are both covered.
    flags={'_paid_allrow_tuning':True}
    if seed is not None:flags['_tuner_random_state']=seed if paid else str(seed)
    elif paid:flags['_tuner_random_state']=None
    adapter=load_adapter(stage)
    model=adapter._make_fedot(SimpleNamespace(type='classification',seed=104735,framework_params=flags),
        'neg_log_loss',dict(n_jobs=8,cv_folds=3,with_tuning=True,use_operations_cache=False,
        use_preprocessing_cache=False,use_predictions_cache=False,final_fit_use_remaining_time=False),54,5.4)
    composer=model.api_composer
    assert getattr(composer,'_experimental_tuner_random_state',None)==seed
    assert hasattr(composer,'_experimental_tuner_random_state')==(seed is not None)
    composer.timer=ApiTime(time_for_automl=54,with_tuning=True)
    model.params.init_params_for_composing(composer.timer.timedelta_composing,multi_objective=False)
    model._fit_started_at=time.monotonic()
    data=InputData(idx=np.arange(18),features=np.arange(108).reshape(18,6),target=np.arange(18)%3,
                   task=Task(TaskTypesEnum.classification),data_type=DataTypesEnum.table)
    if paid:
        pipe=Pipeline(PipelineNode('catboost',nodes_from=[PipelineNode('scaling')]))
        if smoke:pipe.root_node.parameters={**pipe.root_node.parameters,'num_trees':12}
        pipe.replace_n_jobs_in_nodes(8)
    else:
        pipe=graph_from_record(original_graph)
    initial=graph_record(pipe);history=OptHistory();calls=[];requests=[];fmin=[];builders=[];fits=[]
    real_rng=np.random.default_rng
    def observed_rng(value=None):
        if value is None:
            assert seed is None
            requests.append(None);return real_rng(104753+len(requests))
        return real_rng(value)
    def objective(evaluator,pipeline):
        graph=graph_record(pipeline)
        value=(1. if graph[0]['params']['use_eval_set'] else .9) if paid else analytic_score(graph)
        calls.append(dict(graph=graph,fitness=value));return SingleObjFitness(value)
    def forbidden(*args,**kwargs):fits.append(True);raise AssertionError('H26 forbids model fits')
    real_build=TunerBuilder.build
    def observed_build(builder,data):
        tuner=real_build(builder,data)
        builders.append(dict(seed=builder.additional_params.get('random_state'),
            generator=tuner._hyperopt_rstate is not None,tuner_class=type(tuner).__name__))
        return tuner
    def observed_fmin(original,location):
        def call(*args,**kwargs):
            fmin.append(dict(location=location,seeded=kwargs.get('rstate') is not None,max_evals=kwargs['max_evals']))
            return original(*args,**kwargs)
        return call
    before=time.monotonic()
    with ExitStack() as context:
        context.enter_context(patch.dict(os.environ,{},clear=False));os.environ.pop('HYPEROPT_FMIN_SEED',None)
        context.enter_context(patch.object(api_composer,'DEFAULT_TUNING_ITERATIONS_NUMBER',4))
        context.enter_context(patch.object(np.random,'default_rng',observed_rng))
        context.enter_context(patch.object(TunerBuilder,'build',observed_build))
        context.enter_context(patch.object(PipelineObjectiveEvaluate,'evaluate',objective))
        context.enter_context(patch.object(Fedot,'fit',forbidden));context.enter_context(patch.object(Pipeline,'fit',forbidden))
        for module in [simultaneous,hyperopt_tuner]:
            context.enter_context(patch.object(module,'fmin',observed_fmin(module.fmin,module.__name__)))
        random.seed(104735);np.random.seed(104735)
        returned=composer.tune_final_pipeline(data,pipe,history)
    assert not fits and len(builders)==1 and builders[0]['seed']==seed and builders[0]['generator']==(seed is not None)
    assert len(calls)==(3 if paid else 6) and len(fmin)==(1 if paid else 2)
    assert all(c['seeded']==(seed is not None) for c in fmin)
    assert len(requests)==(0 if seed is not None else 1 if paid else 2)
    policy=composer._experimental_paid_policy.record
    assert policy['status']==('evaluated' if paid else 'ineligible_original_path')
    history.save(str(output/'history.json'))
    record=dict(name=name,seed=seed,paid=paid,initial=initial,calls=calls,fmin=fmin,
        builders=builders,unseeded_requests=len(requests),fits=0,policy=policy,
        returned_graph=graph_record(returned),history=canonical_history(history),seconds=time.monotonic()-before,
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(output/'result.json',record);return record


def audit(stage):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    count=verify(stage);records={}
    for name in CELLS:
        d=stage/'cells'/name;r=json.loads((d/'result.json').read_text());p=json.loads((d/'process.json').read_text())
        paid=name.startswith('paid-');seed=104758 if name.endswith('-seeded') else None
        assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=60
        assert r['name']==name and r['seed']==seed and r['paid']==paid and r['fits']==0
        assert len(r['builders'])==1 and r['builders'][0]['seed']==seed and r['builders'][0]['generator']==(seed is not None)
        assert len(r['fmin'])==(1 if paid else 2) and all(v['seeded']==(seed is not None) for v in r['fmin'])
        assert r['unseeded_requests']==(0 if seed is not None else 1 if paid else 2)
        assert r['history']==canonical_history(OptHistory.load(str(d/'history.json')))
        assert r['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
        assert r['environment']['PYTHONPATH']==f'{stage}/vendor:{stage}/source:{stage}/harness'
        if paid:
            assert len(r['calls'])==3 and [c['fitness'] for c in r['calls']]==[1.,.9,.9]
            proposal=deepcopy(r['initial']);proposal[0]['params']['use_eval_set']=False
            assert [c['graph'] for c in r['calls']]==[r['initial'],proposal,proposal]
            assert r['returned_graph']==proposal and proposal[0]['params']['num_trees']==3000
            policy=r['policy'];assert policy['status']=='evaluated' and policy['evaluations_count']==1
            assert policy['admission']['admitted'] and policy['admission']['remaining']>=110
            assert policy['calls'][1]['cap']==policy['calls'][2]['cap']==5
            assert policy['calls'][0]['cap']==300
            assert policy['bounds']['refit_reserve']==324 and policy['bounds']['postfit_reserve']==162
            assert policy['charged_tuning_seconds']<=policy['bounds']['allocation']<=1100
            assert len(r['history'])==3
            assert [g['individuals'][0]['graph'] for g in r['history']]==[r['initial'],proposal,proposal]
            assert [g['individuals'][0]['fitness'] for g in r['history']]==[1.,.9,.9]
            assert [g['individuals'][0]['parents'] for g in r['history']]==[[],[[0,0]],[[0,0]]]
        else:
            reference=json.loads((stage/('h25-seeded-reference.json' if seed is not None else 'h25-default-reference.json')).read_text())
            assert r['policy']['status']=='ineligible_original_path'
            for key in ['calls','returned_graph','history']:assert r[key]==reference[key],key
        records[name]=r
    for key in ['calls','returned_graph','history']:
        assert records['paid-none'][key]==records['paid-seeded'][key]
    return dict(passed=True,verified_files=count,original_matches_h25=True,paid_seed_neutral_fixed_proposal=True,
        both_routes_receive_seed=True,model_fits=0,
        result_hashes={n:hashlib.sha256((stage/'cells'/n/'result.json').read_bytes()).hexdigest() for n in CELLS},
        scope='Factory/composer opt-in routing; artificial objective, no parent transport/model training/default or fullpair admission')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    start=time.monotonic();(stage/'cells').mkdir()
    for name in CELLS:
        run=stage/'cells'/name;run.mkdir();before=time.monotonic()
        env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
        with (run/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,str(stage/'composer_seed_contract.py'),'worker','--stage',str(stage),'--name',name],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=min(60,270-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=p.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before));assert code==0 and not timed_out
    result=audit(stage);result['stage_seconds']=time.monotonic()-start;assert result['stage_seconds']<=270
    save(stage/'independent-audit.json',result);save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit'])
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--name',choices=CELLS);a=p.parse_args()
    if a.mode=='worker':run_case(a.stage,a.stage/'cells'/a.name,a.name,json.loads((a.stage/'h23-entry.json').read_text())['graph'])
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as e:save(a.stage/'FAILED',dict(error=repr(e)));raise
