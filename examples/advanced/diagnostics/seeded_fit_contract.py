"""H28: fixed-count explicit-seed reproducibility with real synthetic TRAIN CV."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples.advanced.diagnostics.cache_startup_contract import save, verify
from examples.advanced.diagnostics.fallback_rng_contract import canonical_history

CELLS = ('first', 'second')


def synthetic(smoke=False):
    count, seed = (30, 104762) if smoke else (90, 104761)
    y = np.arange(count) % 3
    x = np.random.RandomState(seed).normal(size=(count, 6))
    x[:, 0] += .8*y
    x[:, 1] += .5*(y == 1)
    return x, y


def worker(stage, output, smoke=False):
    import pandas as pd
    from fedot.api.api_utils import api_composer
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.tuning import simultaneous, hyperopt_tuner
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record

    x, y = synthetic(smoke)
    frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])
    with (output/'input.npz').open('xb') as f:
        np.savez_compressed(f, features=x, target=y)
    adapter = load_adapter(stage)
    config = SimpleNamespace(type='classification', seed=104735, framework_params={
        '_paid_allrow_tuning': True, '_tuner_random_state': 104760,
        '_paid_policy_observation_dir': str(output/'observation')})
    model = adapter._make_fedot(config, 'neg_log_loss', dict(n_jobs=8, cv_folds=3,
        with_tuning=True, use_operations_cache=False, use_preprocessing_cache=False,
        use_predictions_cache=False, final_fit_use_remaining_time=False), .9, .09)
    node = PipelineNode('logit', nodes_from=[PipelineNode('scaling')])
    node.parameters = {**node.parameters, 'C': 1.}
    graph = Pipeline(node); graph.replace_n_jobs_in_nodes(8)
    model.api_composer.propose_and_fit_initial_assumption = lambda data: ([deepcopy(graph)], deepcopy(graph))
    model.api_composer.compose_pipeline = lambda data, *args: (
        deepcopy(graph), [deepcopy(graph)], SimpleNamespace(history=OptHistory()))
    fits, builders, fmin_calls, rng_ids = [], [], [], []
    fit = Pipeline.fit
    build = TunerBuilder.build

    def observed_fit(pipeline, *args, **kwargs):
        fits.append(graph_record(pipeline))
        return fit(pipeline, *args, **kwargs)

    def observed_build(builder, data):
        tuner = build(builder, data)
        builders.append(dict(seed=builder.additional_params.get('random_state'),
            generator=tuner._hyperopt_rstate is not None))
        return tuner

    def observed_fmin(original, location):
        def call(*args, **kwargs):
            rng = kwargs.get('rstate')
            assert isinstance(rng, np.random.Generator)
            rng_ids.append(id(rng))
            digest = lambda: hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
            record = dict(location=location, max_evals=kwargs['max_evals'], before=digest())
            answer = original(*args, **kwargs)
            record['after'] = digest(); fmin_calls.append(record)
            return answer
        return call

    before = time.monotonic()
    with ExitStack() as context:
        context.enter_context(patch.object(api_composer, 'DEFAULT_TUNING_ITERATIONS_NUMBER', 4))
        context.enter_context(patch.object(Pipeline, 'fit', observed_fit))
        context.enter_context(patch.object(TunerBuilder, 'build', observed_build))
        for module in [simultaneous, hyperopt_tuner]:
            context.enter_context(patch.object(module, 'fmin', observed_fmin(module.fmin, module.__name__)))
        model.fit(frame, y)
    fit_seconds = time.monotonic()-before
    probabilities = model.predict_proba(frame, probs_for_all_classes=True)
    with (output/'train-predictions.npz').open('xb') as f:
        np.savez_compressed(f, probs=probabilities, truth=y, rows=np.arange(len(y)))
    record = dict(smoke=smoke, seed=104760, model_seed=104735, initial_graph=graph_record(graph),
        final_graph=graph_record(model.current_pipeline), history=canonical_history(model.history),
        fits=len(fits), builders=builders, fmin=fmin_calls,
        same_rng=len(rng_ids)==2 and rng_ids[0]==rng_ids[1], fit_seconds=fit_seconds,
        environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']})
    save(output/'result.json', record)
    return record


def audit_cell(output, smoke=False):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from examples.advanced.diagnostics.audit_paid_policy_pair import audit_observation
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from fedot.core.pipelines.adapters import PipelineAdapter
    read = lambda name: json.loads((output/name).read_text())
    r = read('result.json'); obs = output/'observation'
    assert r['smoke'] == smoke and r['seed'] == 104760 and r['model_seed'] == 104735
    assert r['fit_seconds'] <= 54 and r['fits'] >= 19
    assert r['builders'] == [dict(seed=104760, generator=True)] and r['same_rng']
    assert len(r['fmin']) == 2 and r['fmin'][0]['after'] == r['fmin'][1]['before']
    assert all(c['before'] != c['after'] for c in r['fmin'])
    x, y = synthetic(smoke)
    with np.load(output/'input.npz', allow_pickle=False) as z:
        assert np.array_equal(z['features'], x) and np.array_equal(z['target'], y)
    with np.load(obs/'train-target.npz', allow_pickle=False) as z:
        assert np.array_equal(z['target'], y) and np.array_equal(z['rows'], np.arange(len(y)))
    fit = read('observation/fit.json')
    assert abs(fit['api_seconds']-54) < 1e-8
    assert abs(read('observation/tuning-entry.json')['per_fit_cap']-5.4) < 1e-8
    detail = audit_observation(obs)  # Recompute every fold loss/row/class/split/probability.
    exit_record = read('observation/tuning-exit.json')
    calls = exit_record['evaluations']
    assert len(calls) == 6 and all(c['valid'] for c in calls)
    assert exit_record['policy'] == dict(status='ineligible_original_path')
    assert calls[0]['graph'] == r['initial_graph']
    history = OptHistory.load(str(obs/'history.json'))
    assert canonical_history(history) == r['history']
    starts = [g for g in history.generations if g.label == 'tuning_start']
    ends = [g for g in history.generations if g.label == 'tuning_results']
    trials = [ind for g in history.generations if str(g.label).startswith('tuning_iteration_') for ind in g]
    assert len(starts)==len(ends)==1 and len(trials)==4
    initial, final = starts[0][0], ends[0][0]
    adapter = PipelineAdapter()
    assert graph_record(adapter.restore(initial.graph))==calls[0]['graph']
    assert initial.fitness.value==calls[0]['fitness']
    for ind, c in zip(trials, calls[1:-1]):
        assert graph_record(adapter.restore(ind.graph))==c['graph'] and ind.fitness.value==c['fitness']
        assert [p.uid for p in ind.parent_operator.parent_individuals]==[initial.uid]
    assert [p.uid for p in final.parent_operator.parent_individuals]==[initial.uid]
    selected = calls[-1]['fitness'] <= calls[0]['fitness']*(1-.05/100)
    expected = calls[-1] if selected else calls[0]
    assert r['final_graph']==expected['graph']==graph_record(adapter.restore(final.graph))
    assert final.fitness.value==expected['fitness']
    assert any(c['graph']==calls[-1]['graph'] and c['fitness']==calls[-1]['fitness'] for c in calls[1:-1])
    with np.load(output/'train-predictions.npz', allow_pickle=False) as z:
        p=z['probs']; assert p.shape==(len(y),3) and np.isfinite(p).all()
        assert ((p>=0)&(p<=1)).all() and np.max(np.abs(p.sum(axis=1)-1))<1e-6
        assert np.array_equal(z['truth'],y) and np.array_equal(z['rows'],np.arange(len(y)))
    return dict(history=r['history'], fmin=r['fmin'], initial_graph=r['initial_graph'],
        final_graph=r['final_graph'], calls=[dict(graph=c['graph'],fitness=c['fitness']) for c in calls],
        selected_tuned=selected, valid_cv=len(detail['evaluations']), fits=r['fits'])


def compare_cells(first, second):
    assert first==second, 'Seeded real CV/graphs/history/RNG/selection must match exactly'


def audit(stage):
    count=verify(stage); records={}; seconds={}
    for name in CELLS:
        output=stage/'cells'/name
        p=json.loads((output/'process.json').read_text())
        assert p['exit_code']==0 and not p['timed_out'] and p['seconds']<=90
        r=json.loads((output/'result.json').read_text())
        assert r['environment']==dict(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',FEDOT_OPERATION_DEFAULTS_OVERLAY=None)
        records[name]=audit_cell(output); seconds[name]=p['seconds']
    compare_cells(records['first'],records['second'])
    paths=[Path('train-predictions.npz')]
    paths += [p.relative_to(stage/'cells/first') for p in sorted((stage/'cells/first/observation/cv').glob('*/prediction-*.npz'))]
    assert len(paths)==19
    for path in paths:
        with np.load(stage/'cells/first'/path,allow_pickle=False) as a, np.load(stage/'cells/second'/path,allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files: assert np.array_equal(a[key],b[key]),(str(path),key)
    return dict(passed=True,verified_files=count,identical_probability_arrays=19,
        process_seconds=seconds,cell=records['first'],
        scope='Two fixed-count synthetic TRAIN fits with supplied logit/scaling graph; no live evolution, paid-allrow, model quality gain, timed-search or fullpair admission')


def suite(stage):
    from examples.advanced.diagnostics.run_followup import check_idle
    check_idle();verify(stage)
    save(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
    (stage/'cells').mkdir();before=time.monotonic()
    for name in CELLS:
        output=stage/'cells'/name;output.mkdir();start=time.monotonic()
        env=os.environ.copy();env.pop('FEDOT_OPERATION_DEFAULTS_OVERLAY',None)
        env.update(PYTHONPATH=f'{stage}/vendor:{stage}/source:{stage}/harness',AMLB_PATH=str(stage/'harness'),PYTHONDONTWRITEBYTECODE='1')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='8'
        with (output/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,str(stage/'seeded_fit_contract.py'),'worker','--stage',str(stage),'--output',str(output)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=min(90,210-(time.monotonic()-before)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=p.returncode
        save(output/'process.json',dict(exit_code=code,timed_out=timed_out,seconds=time.monotonic()-start))
        assert code==0 and not timed_out
    report=audit(stage);report['stage_seconds']=time.monotonic()-before;assert report['stage_seconds']<=210
    save(stage/'independent-audit.json',report)
    save(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','suite','audit'])
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.stage,a.output)
    elif a.mode=='audit':print(json.dumps(audit(a.stage),indent=2))
    else:
        try:suite(a.stage)
        except BaseException as error:save(a.stage/'FAILED',dict(error=repr(error)));raise
