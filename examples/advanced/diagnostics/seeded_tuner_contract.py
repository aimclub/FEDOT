"""H25: explicit per-tuner RNG through both actual fmin paths, analytic no-fit test."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_startup_contract import save,verify
from examples.advanced.diagnostics.fallback_rng_contract import analytic_score,canonical_history,graph_from_record

CELLS=('default-control','seed-a-first','seed-a-second','seed-b')


def run_case(stage,output,name,graph):
    from fedot.api.main import Fedot
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.repository.tasks import Task,TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.tuning import simultaneous,hyperopt_tuner
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    assert name in CELLS
    seed=None if name=='default-control' else 104759 if name=='seed-b' else 104758
    calls=[];requests=[];fmin_calls=[];fits=[]
    real_rng=np.random.default_rng
    def tracked_rng(value=None):
        caller=inspect.currentframe().f_back
        requests.append(dict(seed=int(value) if value is not None else None,
            caller=Path(caller.f_code.co_filename).name,function=caller.f_code.co_name))
        if value is None:
            assert seed is None,'Seeded path must never request unseeded entropy'
            return real_rng(104753+sum(r['seed'] is None for r in requests))
        return real_rng(value)
    def objective(evaluator,pipeline):
        g=graph_record(pipeline);value=analytic_score(g)
        calls.append(dict(graph=g,fitness=value));return SingleObjFitness(value)
    def forbid(*args,**kwargs):fits.append(True);raise AssertionError('Model training forbidden')
    rng_ids=[]
    def traced_fmin(original,location):
        def call(*args,**kwargs):
            rng=kwargs.get('rstate');rng_ids.append(id(rng) if rng is not None else None)
            state=lambda:hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest() if rng is not None else None
            record=dict(location=location,max_evals=kwargs['max_evals'],explicit_rstate=rng is not None,before=state())
            answer=original(*args,**kwargs);record['after']=state();fmin_calls.append(record)
            return answer
        return call
    history=OptHistory();task=Task(TaskTypesEnum.classification)
    data=InputData(idx=np.arange(18),features=np.arange(108).reshape(18,6),target=np.arange(18)%3,
                   task=task,data_type=DataTypesEnum.table)
    pipe=graph_from_record(graph);assert graph_record(pipe)==graph
    before=time.monotonic()
    with ExitStack() as context:
        context.enter_context(patch.dict(os.environ,{},clear=False));os.environ.pop('HYPEROPT_FMIN_SEED',None)
        context.enter_context(patch.object(np.random,'default_rng',tracked_rng))
        context.enter_context(patch.object(PipelineObjectiveEvaluate,'evaluate',objective))
        context.enter_context(patch.object(Fedot,'fit',forbid));context.enter_context(patch.object(Pipeline,'fit',forbid))
        for module in [simultaneous,hyperopt_tuner]:
            context.enter_context(patch.object(module,'fmin',traced_fmin(module.fmin,module.__name__)))
        random.seed(104735);np.random.seed(104735)
        builder=(TunerBuilder(task).with_tuner(simultaneous.SimultaneousTuner)
            .with_metric(ClassificationMetricsEnum.logloss).with_cv_folds(3).with_n_jobs(8)
            .with_iterations(4).with_timeout(timedelta(seconds=30)).with_history(history))
        if seed is not None:builder.with_additional_params(random_state=seed)
        tuner=builder.build(data)
        returned=tuner.tune(pipe,show_progress=False)
    assert not fits and len(calls)==6 and len(fmin_calls)==2
    assert sum(r['seed'] is None for r in requests)==(2 if seed is None else 0)
    roots=[r for r in requests if r['caller']=='hyperopt_tuner.py' and r['function']=='__init__']
    assert [r['seed'] for r in roots]==([] if seed is None else [seed])
    assert all(c['explicit_rstate']==(seed is not None) for c in fmin_calls)
    same_rng=seed is not None and rng_ids[0]==rng_ids[1]
    if seed is not None:
        assert same_rng and fmin_calls[0]['after']==fmin_calls[1]['before']
        assert all(c['before']!=c['after'] for c in fmin_calls)
    history.save(str(output/'history.json'))
    r=dict(name=name,seed=seed,calls=calls,requests=requests,fmin=fmin_calls,same_rng=same_rng,
        returned_graph=graph_record(returned),history=canonical_history(history),fits=0,
        seconds=time.monotonic()-before,environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(output/'result.json',r)
    return r


def compare(records,reference):
    a,b=records['seed-a-first'],records['seed-a-second']
    for k in ['calls','returned_graph','history','requests','fmin']:assert a[k]==b[k],k
    assert a['calls'][1:5]!=records['seed-b']['calls'][1:5]
    for k in ['calls','returned_graph','history']:assert records['default-control'][k]==reference[k],k
    return dict(passed=True,same_seed_exact=True,different_seed_trials_differ=True,
        default_matches_frozen_h24=True,model_fits=0,
        scope='Analytic per-instance seeded tuner capability; no composer/adapter seed integration, quality or default promotion')


def audit(stage):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    count=verify(stage);records={}
    graph=json.loads((stage/'h23-entry.json').read_text())['graph']
    for name in CELLS:
        d=stage/'cells'/name;r=json.loads((d/'result.json').read_text());p=json.loads((d/'process.json').read_text())
        assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=60
        seed=None if name=='default-control' else 104759 if name=='seed-b' else 104758
        assert r['name']==name and r['seed']==seed and r['fits']==0 and len(r['calls'])==6
        assert r['calls'][0]['graph']==graph
        assert sum(c['seed'] is None for c in r['requests'])==(2 if seed is None else 0)
        roots=[c for c in r['requests'] if c['caller']=='hyperopt_tuner.py' and c['function']=='__init__']
        assert [c['seed'] for c in roots]==([] if seed is None else [seed])
        assert len(r['fmin'])==2 and all(c['explicit_rstate']==(seed is not None) for c in r['fmin'])
        if seed is not None:
            assert r['same_rng'] and r['fmin'][0]['after']==r['fmin'][1]['before']
            assert all(c['before']!=c['after'] for c in r['fmin'])
        for call in r['calls']:assert call['fitness']==analytic_score(call['graph'])
        assert r['history']==canonical_history(OptHistory.load(str(d/'history.json')))
        selected=r['calls'][-1]['fitness']<=r['calls'][0]['fitness']*(1-.05/100)
        expected=r['calls'][-1] if selected else r['calls'][0]
        assert r['returned_graph']==expected['graph']
        assert r['history'][-1]['individuals'][0]['fitness']==expected['fitness']
        assert r['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
        assert r['environment']['PYTHONPATH']==f'{stage}/vendor:{stage}/source:{stage}/harness'
        records[name]=r
    result=compare(records,json.loads((stage/'h24-default-reference.json').read_text()))
    result['verified_files']=count
    result['result_hashes']={name:hashlib.sha256((stage/'cells'/name/'result.json').read_bytes()).hexdigest() for name in CELLS}
    return result


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
            p=subprocess.Popen([sys.executable,str(stage/'seeded_tuner_contract.py'),'worker','--stage',str(stage),'--name',name],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=min(60,270-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=p.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before))
        assert code==0 and not timed_out
    result=audit(stage);result['stage_seconds']=time.monotonic()-start;assert result['stage_seconds']<=270
    save(stage/'independent-audit.json',result)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit'])
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--name',choices=CELLS);a=p.parse_args()
    if a.mode=='worker':run_case(a.stage,a.stage/'cells'/a.name,a.name,json.loads((a.stage/'h23-entry.json').read_text())['graph'])
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as e:save(a.stage/'FAILED',dict(error=repr(e)));raise
