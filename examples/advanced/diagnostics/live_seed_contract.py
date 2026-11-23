"""H29: conditional flag neutrality after real evolution with seeded fixed-count tuning."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics import live_adapter_contract as lifecycle
from examples.advanced.diagnostics.cache_startup_contract import save, verify, prepare_run

SIDES = lifecycle.SIDES
BASE_SETTINGS = lifecycle.settings


def synthetic():
    rng = np.random.RandomState(104763)
    y = np.arange(120) % 3
    x = rng.normal(size=(120, 6)).astype('float32')
    x[:, 0] += 2*y
    x[:, 1] += y == 1
    x[:, 4:] = np.arange(240).reshape(120, 2) % 4
    return x, y


def settings(run, enabled):
    params = BASE_SETTINGS(run, enabled)
    params['_tuner_random_state'] = 104760
    return params


@contextmanager
def trace_tuner(record):
    from fedot.api.api_utils import api_composer
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from golem.core.tuning import hyperopt_tuner, simultaneous
    real_build = TunerBuilder.build
    identities = []

    def build(builder, data):
        tuner = real_build(builder, data)
        record['builders'].append(dict(seed=builder.additional_params.get('random_state'),
            generator=tuner._hyperopt_rstate is not None))
        return tuner

    def observed(original, location):
        def call(*args, **kwargs):
            rng = kwargs.get('rstate'); identities.append(id(rng))
            assert isinstance(rng, np.random.Generator)
            digest = lambda: hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
            item = dict(location=location, max_evals=kwargs['max_evals'], before=digest(), succeeded=False)
            try:
                answer = original(*args, **kwargs); item['succeeded'] = True
                return answer
            finally:
                item['after'] = digest(); record['fmin'].append(item)
        return call

    with ExitStack() as context:
        context.enter_context(patch.object(api_composer, 'DEFAULT_TUNING_ITERATIONS_NUMBER', 4))
        context.enter_context(patch.object(TunerBuilder, 'build', build))
        for module in [hyperopt_tuner, simultaneous]:
            context.enter_context(patch.object(module, 'fmin', observed(module.fmin, module.__name__)))
        try: yield
        finally: record['same_rng'] = bool(identities) and len(set(identities)) == 1


def child(dataset, config):
    stage=Path(config.contract_stage);run=stage/'runs'/config.contract_side
    assert config.framework_params['_tuner_random_state']==104760 and config.seed==104735
    record=dict(builders=[],fmin=[],tuner_seed=104760,model_seed=config.seed)
    try:
        with patch.object(lifecycle,'synthetic',synthetic), trace_tuner(record):
            return lifecycle.child(dataset,config)
    finally: save(run/'rng.json',record)


def redirect_child(real, stage):
    def redirect(caller_file, script_file, *args, **kwargs):
        assert script_file==str(stage/'live_adapter_contract.py')
        return real(caller_file,str(stage/'live_seed_contract.py'),*args,**kwargs)
    return redirect


def parent(stage, side, framework_python):
    from frameworks.shared import caller
    with patch.object(caller,'run_in_venv',redirect_child(caller.run_in_venv,stage)), \
            patch.object(lifecycle,'synthetic',synthetic), patch.object(lifecycle,'settings',settings):
        lifecycle.parent(stage,side,framework_python)


def fallback_history(directory):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    read=lambda n:json.loads((directory/n).read_text())
    calls=read('tuning-exit.json')['evaluations']
    assert len(calls)>=3 and all(c['valid'] for c in calls)
    h=OptHistory.load(str(directory/'history.json'));adapter=PipelineAdapter()
    starts=[g for g in h.generations if g.label=='tuning_start']
    ends=[g for g in h.generations if g.label=='tuning_results']
    trials=[ind for g in h.generations if str(g.label).startswith('tuning_iteration_') for ind in g]
    assert len(starts)==len(ends)==1 and len(trials)==len(calls)-2
    initial,final=starts[0][0],ends[0][0]
    assert graph_record(adapter.restore(initial.graph))==calls[0]['graph']
    assert initial.fitness.value==calls[0]['fitness']
    for ind,call in zip(trials,calls[1:-1]):
        assert graph_record(adapter.restore(ind.graph))==call['graph'] and ind.fitness.value==call['fitness']
        assert [p.uid for p in ind.parent_operator.parent_individuals]==[initial.uid]
    assert [p.uid for p in final.parent_operator.parent_individuals]==[initial.uid]
    selected=calls[-1]['fitness']<=calls[0]['fitness']*(1-.05/100)
    expected=calls[-1] if selected else calls[0]
    returned=read('tuning-return.json')['graph']
    assert returned==expected['graph']==graph_record(adapter.restore(final.graph))
    assert final.fitness.value==expected['fitness']
    assert any(c['graph']==calls[-1]['graph'] and c['fitness']==calls[-1]['fitness'] for c in calls[1:-1])
    return dict(calls=[dict(graph=c['graph'],fitness=c['fitness']) for c in calls],
        labels=[g.label for g in h.generations if str(g.label).startswith('tuning_')],
        parents='each trial and final has initial tuning UID as sole parent',
        returned=returned,returned_fitness=final.fitness.value,selected=selected)


def applicability(a,b,policy,calls):
    if a!=b:return 'different_composed_graphs'
    if policy!={'status':'ineligible_original_path'}:return 'candidate_not_original_fallback'
    if any(len(group)!=6 or not all(c['valid'] for c in group) for group in calls):
        return 'fixed_six_valid_evaluations_not_completed'
    return None


def audit(stage):
    # Inherited lifecycle audit independently recomputes CV and validates actual
    # composition, predictions, final refit, API/process budgets and caches.
    with patch.object(lifecycle,'synthetic',synthetic): base=lifecycle.audit(stage)
    dirs={side:stage/'runs'/side/'observation' for side in SIDES}
    entries={s:json.loads((d/'tuning-entry.json').read_text()) for s,d in dirs.items()}
    exits={s:json.loads((d/'tuning-exit.json').read_text()) for s,d in dirs.items()}
    traces={s:json.loads((stage/'runs'/s/'rng.json').read_text()) for s in SIDES}
    for s,r in traces.items():
        assert r['tuner_seed']==104760 and r['model_seed']==104735
        assert r['builders']==[dict(seed=104760,generator=True)]
        assert r['same_rng'] and all(c['succeeded'] for c in r['fmin'])
        assert all(c['before']!=c['after'] for c in r['fmin'])
        if len(r['fmin'])==2:assert r['fmin'][0]['after']==r['fmin'][1]['before']
    a,b=SIDES
    reason=applicability(entries[a]['graph'],entries[b]['graph'],exits[b]['policy'],
        [exits[s]['evaluations'] for s in SIDES])
    lineage={}
    for s in SIDES:
        calls=exits[s]['evaluations'];policy=exits[s]['policy']
        if policy in [None,{'status':'ineligible_original_path'}] and len(calls)>=3 and all(c['valid'] for c in calls):
            lineage[s]=fallback_history(dirs[s])
    gates={}
    if reason is None:
        gates['tuning_history_and_fitness_equal']=lineage[a]==lineage[b]
        gates['rng_trace_equal']=traces[a]==traces[b] and len(traces[a]['fmin'])==2
        files=[p.relative_to(dirs[a]) for p in sorted((dirs[a]/'cv').glob('*/prediction-*.npz'))]
        assert len(files)==18
        equality=True
        for path in files:
            with np.load(dirs[a]/path,allow_pickle=False) as x,np.load(dirs[b]/path,allow_pickle=False) as y:
                equality &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        gates['all_18_cv_arrays_equal']=bool(equality)
        with np.load(stage/'runs'/a/'synthetic-predictions.npz',allow_pickle=False) as x, \
                np.load(stage/'runs'/b/'synthetic-predictions.npz',allow_pickle=False) as y:
            gates['synthetic_prediction_arrays_equal']=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
    return dict(lifecycle_passed=True,verified_files=base['verified_files'],applicable=reason is None,
        inapplicable_reason=reason,mechanism_passed=reason is None and all(gates.values()),gates=gates,
        sides=base['sides'],fallback_histories=lineage,rng_traces=traces,
        scope='Conditional ineligible-flag neutrality after bounded live evolution, common explicit seed and fixed-count tuning; synthetic data only, no quality gain/timed-search/default/fullpair admission')


def suite(stage, framework_python, benchmark_python):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    (stage/'runs').mkdir();start=time.monotonic()
    for side in SIDES:
        run=prepare_run(stage/'runs',side);before=time.monotonic()
        env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='8'
        with (run/'worker.log').open('x') as log:
            p=subprocess.Popen([benchmark_python,str(stage/'live_seed_contract.py'),'parent','--stage',str(stage),
                '--side',side,'--framework-python',framework_python],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=min(330,670-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=p.returncode
        save(run/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before))
        assert code==0 and not timed_out
    report=audit(stage);report['stage_seconds']=time.monotonic()-start;assert report['stage_seconds']<=670
    save(stage/'independent-audit.json',report)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['child','parent','suite','audit'])
    p.add_argument('--stage',type=Path);p.add_argument('--side',choices=SIDES)
    p.add_argument('--framework-python');p.add_argument('--benchmark-python');a=p.parse_args()
    if a.mode=='child':
        from frameworks.shared.callee import call_run
        call_run(child)
    elif a.mode=='parent':parent(a.stage,a.side,a.framework_python)
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage,a.framework_python,a.benchmark_python)
        except BaseException as error:save(a.stage/'FAILED',dict(error=repr(error)));raise
