"""H24: ineligible-hook neutrality under controlled Hyperopt entropy; no model fits."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_startup_contract import save, verify

CELLS=('a-false','a-true','b-false','b-true')


def analytic_score(graph):
    """Deterministic artificial scalar, NOT measured model loss or borrowed fitness."""
    payload=json.dumps(graph,sort_keys=True,separators=(',',':')).encode()
    return .5+int(hashlib.sha256(payload).hexdigest()[:12],16)/float(16**12)


def rng_digest():
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()))).hexdigest()


def canonical_history(history):
    from fedot.core.pipelines.adapters import PipelineAdapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    ids={str(ind.uid):(i,j) for i,g in enumerate(history.generations) for j,ind in enumerate(g)}
    records=[]
    for g in history.generations:
        individuals=[]
        for ind in g:
            parent=ind.parent_operator
            individuals.append(dict(graph=graph_record(PipelineAdapter().restore(ind.graph)),
                fitness=float(ind.fitness.value) if ind.fitness.valid else None,
                parents=[list(ids[str(p.uid)]) for p in parent.parent_individuals] if parent else []))
        records.append(dict(label=g.label,individuals=individuals))
    return records


def graph_from_record(record):
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    nodes={}
    def build(i):
        if i not in nodes:
            item=record[i]
            node=PipelineNode(item['operation'],nodes_from=[build(p) for p in item['parents']] or None)
            node.parameters=deepcopy(item['params']);nodes[i]=node
        return nodes[i]
    return Pipeline(build(0))


def run_case(stage, output, enabled, stream_seed, graph):
    from fedot.api.main import Fedot
    from fedot.api.time import ApiTime
    from fedot.api.api_utils import api_composer
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.tuning import simultaneous,hyperopt_tuner
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.composer_paid_policy import ComposerPaidPolicy
    adapter=load_adapter(stage)
    model=adapter._make_fedot(SimpleNamespace(type='classification',seed=104735,
        framework_params={'_paid_allrow_tuning':enabled}), 'neg_log_loss',
        dict(n_jobs=8,cv_folds=3,with_tuning=True,use_operations_cache=False,
             use_preprocessing_cache=False,use_predictions_cache=False,
             final_fit_use_remaining_time=False),3.6,.36)
    composer=model.api_composer
    composer.timer=ApiTime(time_for_automl=3.6,with_tuning=True)
    model.params.init_params_for_composing(composer.timer.timedelta_composing,multi_objective=False)
    data=InputData(idx=np.arange(18),features=np.arange(108).reshape(18,6),target=np.arange(18)%3,
        task=Task(TaskTypesEnum.classification),data_type=DataTypesEnum.table)
    pipe=graph_from_record(graph);assert graph_record(pipe)==graph
    history=OptHistory();calls=[];entropy=[];fmin_calls=[];hook=[];fits=[]
    real_rng=np.random.default_rng
    def controlled_rng(seed=None):
        if seed is None:
            assigned=stream_seed+len(entropy)
            entropy.append(dict(requested_seed=None,assigned_seed=assigned))
            return real_rng(assigned)
        return real_rng(seed)
    def analytic(evaluator,pipeline):
        record=graph_record(pipeline);value=analytic_score(record)
        calls.append(dict(graph=record,fitness=value))
        return SingleObjFitness(value)
    def forbid(*args,**kwargs):
        fits.append(True);raise AssertionError('H24 forbids model training')
    original_policy=ComposerPaidPolicy.__call__
    def observe_hook(policy,*args,**kwargs):
        before=rng_digest();result=original_policy(policy,*args,**kwargs)
        hook.append(dict(before=before,after=rng_digest(),not_implemented=result is NotImplemented))
        assert hook[-1]['before']==hook[-1]['after'] and result is NotImplemented
        return result
    def observe_fmin(original,location):
        def call(*args,**kwargs):
            fmin_calls.append(dict(location=location,max_evals=kwargs['max_evals'],
                explicit_rstate=kwargs.get('rstate') is not None))
            return original(*args,**kwargs)
        return call
    before=time.monotonic()
    with ExitStack() as context:
        context.enter_context(patch.dict(os.environ,{},clear=False))
        os.environ.pop('HYPEROPT_FMIN_SEED',None)
        context.enter_context(patch.object(np.random,'default_rng',controlled_rng))
        context.enter_context(patch.object(ComposerPaidPolicy,'__call__',observe_hook))
        context.enter_context(patch.object(api_composer,'DEFAULT_TUNING_ITERATIONS_NUMBER',4))
        for module in [simultaneous,hyperopt_tuner]:
            context.enter_context(patch.object(module,'fmin',observe_fmin(module.fmin,module.__name__)))
        context.enter_context(patch.object(PipelineObjectiveEvaluate,'evaluate',analytic))
        context.enter_context(patch.object(Fedot,'fit',forbid))
        context.enter_context(patch.object(Pipeline,'fit',forbid))
        random.seed(104735);np.random.seed(104735)
        returned=composer.tune_final_pipeline(data,pipe,history)
    assert len(calls)==6 and len(entropy)==2 and len(fmin_calls)==2 and not fits
    assert all(not c['explicit_rstate'] for c in fmin_calls)
    assert len(hook)==int(enabled)
    policy=getattr(composer,'_experimental_paid_policy',None)
    assert (policy.record if policy else None)==({'status':'ineligible_original_path'} if enabled else None)
    history.save(str(output/'history.json'))
    record=dict(enabled=enabled,stream_seed=stream_seed,calls=calls,entropy=entropy,fmin=fmin_calls,
        hook=hook,fits=len(fits),returned_graph=graph_record(returned),history=canonical_history(history),
        policy=policy.record if policy else None,seconds=time.monotonic()-before,
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(output/'result.json',record)
    return record


def compare(records):
    for stream in ['a','b']:
        a,b=[records[f'{stream}-{flag}'] for flag in ['false','true']]
        for key in ['calls','entropy','fmin','returned_graph','history']:
            assert a[key]==b[key],(stream,key)
        assert a['fits']==b['fits']==0
        assert len(a['calls'])==len(b['calls'])==6
    assert records['a-false']['calls'][1:5]!=records['b-false']['calls'][1:5]
    return dict(passed=True,within_stream_exact=True,between_stream_trials_differ=True,model_fits=0,
        scope='Analytic no-fit objective; controlled entropy at default_rng boundary; no historical unique-cause or model-quality claim')


def audit(stage):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    count=verify(stage);records={}
    for name in CELLS:
        run=stage/'cells'/name
        r=json.loads((run/'result.json').read_text());p=json.loads((run/'process.json').read_text())
        assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=60
        assert r['enabled']==name.endswith('-true') and r['stream_seed']==(104754 if name[0]=='a' else 104756)
        assert r['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
        assert r['environment']['PYTHONPATH']==f'{stage}/vendor:{stage}/source:{stage}/harness'
        assert r['history']==canonical_history(OptHistory.load(str(run/'history.json')))
        assert len(r['entropy'])==2 and len(r['fmin'])==2
        assert all(not c['explicit_rstate'] for c in r['fmin'])
        for i,e in enumerate(r['entropy']):assert e==dict(requested_seed=None,assigned_seed=r['stream_seed']+i)
        assert len(r['hook'])==int(r['enabled'])
        for event in r['hook']:assert event['before']==event['after'] and event['not_implemented']
        for c in r['calls']:assert c['fitness']==analytic_score(c['graph'])
        records[name]=r
    result=compare(records);result.update(verified_files=count,
        result_hashes={name:hashlib.sha256((stage/'cells'/name/'result.json').read_bytes()).hexdigest() for name in CELLS})
    return result


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    start=time.monotonic();(stage/'cells').mkdir()
    for name in CELLS:
        run=stage/'cells'/name;run.mkdir()
        before=time.monotonic();env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='8'
        with (run/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,str(stage/'fallback_rng_contract.py'),'worker','--stage',str(stage),'--name',name],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=min(60,270-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=p.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before))
        assert code==0 and not timed_out
    result=audit(stage);result['stage_seconds']=time.monotonic()-start
    assert result['stage_seconds']<=270
    save(stage/'independent-audit.json',result)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit'])
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--name',choices=CELLS)
    a=p.parse_args()
    if a.mode=='worker':
        run_case(a.stage,a.stage/'cells'/a.name,a.name.endswith('-true'),104754 if a.name[0]=='a' else 104756,
                 json.loads((a.stage/'h23-entry.json').read_text())['graph'])
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as e:save(a.stage/'FAILED',dict(error=repr(e)));raise
