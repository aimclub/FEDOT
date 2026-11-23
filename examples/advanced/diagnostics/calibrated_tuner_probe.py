"""H17: opt-in paid calibration in the real TunerBuilder/SimultaneousTuner.

Diagnostic only. No composer/default integration and no outer data access.
"""
import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np

from examples.advanced.diagnostics.cache_cost_probe import probability_loss, sha, write_json

SEED = 104735
SPLIT_SEED = 42  # Actual TunerBuilder default, not a replacement data splitter.
POLICY_SECONDS = 1100


def calibrated_cap(seconds, ceiling=300):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Calibration must be finite and positive')
    # The real evaluator rejects a fold with <=1s remaining. A tiny calibration
    # must not create an unusable sub-second cap. Full fabert caps exceed 200s.
    return min(ceiling, max(5., 1.3 * seconds))


def admitted(remaining, cap, overhead=50):
    return remaining >= 2 * (cap + overhead)


def parameter_space(pipeline):
    """All parameters, since BaseTuner.set_arg_node REPLACES its dictionary."""
    return fixed_space(graph_record(pipeline))


def fixed_space(nodes):
    from hyperopt import hp
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
    operations = {}
    for node in nodes:
        name = node['operation']
        if name in operations:
            raise ValueError('This fixed protocol requires unique operation types')
        params = deepcopy(node['params'])
        if name == 'catboost':
            params['use_eval_set'] = False
        operations[name] = {k: {'hyperopt-dist': hp.choice, 'sampling-scope': [[v]],
                                'type': 'categorical'} for k, v in params.items()}
    return PipelineSearchSpace(operations, replace_default_search_space=True)


def make_tuner_class():
    from golem.core.tuning.simultaneous import SimultaneousTuner

    class PaidCalibrationTuner(SimultaneousTuner):
        def __init__(self, objective_evaluate, deadline, recorder, ceiling=300,
                     overhead=50, **kwargs):
            self.deadline, self.recorder = deadline, recorder
            self.ceiling, self.overhead = ceiling, overhead
            self.cv_evaluator = objective_evaluate
            self.cap = ceiling
            self.calls = []
            self.admission = None
            self.allow_trial = False

            def observed(graph):
                phase = ('initial', 'proposal', 'confirmation')[len(self.calls)]
                remaining = self.deadline - time.monotonic()
                if remaining <= 1:
                    raise TimeoutError('No whole-policy time remaining')
                cap = min(self.cap, max(0.001, remaining - 5))
                self.cv_evaluator._evaluation_time_constraint = timedelta(seconds=cap)
                random.seed(SEED)
                np.random.seed(SEED)
                fitness, seconds = self.recorder(self.cv_evaluator, graph, phase, cap)
                self.calls.append(dict(phase=phase, valid=bool(fitness.valid), seconds=seconds,
                                       cap=cap, fitness=float(fitness.value) if fitness.valid else None))
                return fitness

            super().__init__(objective_evaluate=observed, **kwargs)

        def init_check(self, graph):
            super().init_check(graph)  # Actual native initial check and history.
            initial = self.calls[0]
            if initial['valid']:
                self.cap = calibrated_cap(initial['seconds'], self.ceiling)

        def _tune(self, graph, **kwargs):
            remaining = self.deadline - time.monotonic()
            self.allow_trial = self.calls[0]['valid'] and admitted(remaining, self.cap, self.overhead)
            self.admission = dict(remaining=remaining, cap=self.cap, overhead=self.overhead,
                                  admitted=self.allow_trial)
            if not self.allow_trial:
                return graph
            return super()._tune(graph, **kwargs)  # Real Hyperopt, ONE fixed proposal.

        def final_check(self, graphs, multi_obj=False):
            if not self.allow_trial:
                # Do not waste another CV after invalid calibration/no reserve.
                self.obtained_metric = self.init_metric if self.calls[0]['valid'] else None
                self._record_tuning_results([self.init_graph], [self.obtained_metric])
                return self.init_graph
            return super().final_check(graphs, multi_obj)  # Real final CV/selection/history.

    return PaidCalibrationTuner


def graph_record(graph):
    return [dict(operation=n.operation.operation_type, params=deepcopy(n.parameters),
                 parents=[graph.nodes.index(p) for p in (n.nodes_from or [])]) for n in graph.nodes]


def run_tuner(stage, smoke=False):
    started = time.monotonic()
    import pandas as pd
    import fedot
    import golem
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics.adapter_fidelity_probe import capture, summarize
    from examples.advanced.diagnostics.allrow_fit_probe import instrument_fit

    with np.load(stage/'train.npz', allow_pickle=False) as z:
        x, y = z['features'].copy(), z['target'].ravel().copy()
    captured = capture(load_adapter(stage), pd.read_pickle(stage/'train-frame.pkl'), y)
    data = captured['data']
    assert np.array_equal(data.idx, np.arange(len(y)))
    assert np.array_equal(np.asarray(data.target).ravel(), y)
    assert np.array_equal(np.asarray(data.features, dtype=np.float32), x)
    write_json(stage/'input.json', dict(schema=summarize(data), params=captured['params']))
    pipe = Pipeline(PipelineNode('catboost', nodes_from=[PipelineNode('scaling')]))
    if smoke:
        pipe.root_node.parameters = {**pipe.root_node.parameters, 'num_trees': 8, 'max_depth': 2}
    pipe.replace_n_jobs_in_nodes(8)
    initial_graph = graph_record(pipe)
    current = {}
    classes = np.unique(y)

    def metric(graph, reference_data, fold_id=None, **kwargs):
        probs = np.asarray(graph.predict(reference_data).predict).copy()
        truth = np.asarray(reference_data.target).ravel().copy()
        rows = np.asarray(reference_data.idx).copy()
        fitted = []
        for node in graph.nodes:
            model = getattr(node.fitted_operation, 'model', node.fitted_operation)
            if hasattr(model, 'tree_count_'):
                fitted.append(dict(operation=node.operation.operation_type, trees=int(model.tree_count_),
                                   categories=model.get_cat_feature_indices()))
        current['arrays'].update({f'probs_{fold_id}': probs, f'truth_{fold_id}': truth, f'rows_{fold_id}': rows})
        current['folds'].append(dict(fold=fold_id, fitted=fitted))
        return probability_loss(probs, truth, classes)

    def recorder(evaluator, graph, phase, cap):
        current.clear()
        current.update(arrays={'classes': classes}, folds=[])
        assert evaluator._operations_cache is None and evaluator._preprocessing_cache is None and evaluator._predictions_cache is None
        for i, (train, _) in enumerate(evaluator._data_producer()):
            current['arrays'][f'train_rows_{i}'] = np.asarray(train.idx).copy()
        before_graph = graph_record(graph)
        pool_dir = stage/'pool-audits'/phase
        (pool_dir/'pools').mkdir(parents=True)
        original = instrument_fit(pool_dir)
        before = time.monotonic()
        try:
            fitness = evaluator(graph)
        finally:
            FedotCatBoostImplementation.fit = original
        seconds = time.monotonic() - before
        path = stage/'cells'/phase
        with path.with_suffix('.npz').open('xb') as f:
            np.savez_compressed(f, **current['arrays'])
        write_json(path.with_suffix('.json'), dict(valid=bool(fitness.valid), seconds=seconds,
                   fitness=float(fitness.value) if fitness.valid else None, cap=cap, folds=current['folds'],
                   graph_before=before_graph, graph_after=graph_record(graph), npz_sha256=sha(path.with_suffix('.npz'))))
        return fitness, seconds

    history = OptHistory()
    tuner = (TunerBuilder(data.task).with_tuner(make_tuner_class()).with_metric(metric)
             .with_cv_folds(3).with_n_jobs(8).with_iterations(1)
             .with_timeout(timedelta(seconds=POLICY_SECONDS)).with_history(history)
             .with_evaluation_time_constraint(timedelta(seconds=300))
             .with_search_space(parameter_space(pipe))
             .with_additional_params(deadline=started+POLICY_SECONDS, recorder=recorder,
                                     ceiling=300, overhead=50, deviation=1.0).build(data))
    tuned = tuner.tune(pipe, show_progress=False)
    history.save(str(stage/'history.json'))
    history_graphs = [[dict(graph=graph_record(tuner.adapter.restore(ind.graph)),
                           valid=bool(ind.fitness.valid), fitness=float(ind.fitness.value) if ind.fitness.valid else None)
                       for ind in generation] for generation in history.generations]
    result = dict(initial_graph=initial_graph, returned_graph=graph_record(tuned),
                  calls=tuner.calls, admission=tuner.admission, evaluations_count=tuner.evaluations_count,
                  was_tuned=tuner.was_tuned, init_metric=float(tuner.init_metric),
                  obtained_metric=float(tuner.obtained_metric) if tuner.obtained_metric is not None else None, history=history_graphs,
                  history_sha256=sha(stage/'history.json'),
                  policy_seconds=time.monotonic()-started, smoke=smoke,
                  runtime=dict(fedot=fedot.__file__, golem=golem.__file__, python=sys.executable,
                               environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']}))
    write_json(stage/'result.json', result)
    return result


def audit(stage, smoke=False):
    from sklearn.model_selection import StratifiedKFold
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    r = json.loads((stage/'result.json').read_text())
    with np.load(stage/'train.npz', allow_pickle=False) as z:
        y = z['target'].ravel()
    trees = 8 if smoke else 3000
    cats = [2,3] if smoke else list(range(715,800))
    assert len(y) == (60 if smoke else 7413)
    initial = r['initial_graph']
    cat_index = next(i for i,n in enumerate(initial) if n['operation']=='catboost')
    assert len(initial)==2 and initial[cat_index]['params']['num_trees']==trees
    assert initial[cat_index]['params']['use_eval_set'] is True
    assert 'colsample_bylevel' not in initial[cat_index]['params']
    proposal = deepcopy(initial)
    proposal[cat_index]['params']['use_eval_set'] = False
    assert r['runtime']['environment']['FEDOT_OPERATION_DEFAULTS_OVERLAY'] is None
    assert r['policy_seconds'] <= POLICY_SECONDS
    assert sum(c['seconds'] for c in r['calls'])<=r['policy_seconds']
    assert r['history_sha256']==sha(stage/'history.json')
    saved_history=OptHistory.load(str(stage/'history.json'))
    adapter=PipelineAdapter()
    restored=[[dict(graph=graph_record(adapter.restore(ind.graph)),valid=bool(ind.fitness.valid),
                    fitness=float(ind.fitness.value) if ind.fitness.valid else None) for ind in generation]
              for generation in saved_history.generations]
    assert restored==r['history'], 'Serialized history differs from recorded history'
    labels=[getattr(g.label,'value',g.label) for g in saved_history.generations]
    assert labels[0]=='tuning_start' and labels[-1]=='tuning_results'
    root=saved_history.generations[0][0]
    for generation in saved_history.generations[1:]:
        assert all(ind.parent_operator.parent_individuals[0].uid==root.uid for ind in generation)
    if not smoke:
        from examples.advanced.diagnostics.subspace_validation_probe import graph_params
        reference=json.loads((stage/'native-params-reference.json').read_text())
        assert initial[cat_index]['params']==graph_params(reference['graph_id_before'])
        scale=next(n for n in initial if n['operation']=='scaling')
        assert scale['params']=={} and scale['parents']==[]
        assert initial[cat_index]['parents']==[initial.index(scale)]
        assert Path(r['runtime']['fedot']).resolve().is_relative_to((stage/'source').resolve())
        assert Path(r['runtime']['golem']).resolve().is_relative_to((stage/'vendor').resolve())
        assert r['runtime']['environment']['PYTHONPATH']==':'.join(str(stage/p) for p in ['vendor','source','harness'])
        process=json.loads((stage/'process.json').read_text())
        assert process['exit_code']==0 and r['policy_seconds']<=process['seconds']<=1100
        assert process['wait_limit']==1100 and process['kill_grace']==5
    a = r['admission']
    assert a['admitted'] == (r['calls'][0]['valid'] and admitted(a['remaining'],a['cap'],a['overhead']))
    assert a['overhead']==50
    assert a['cap']==(calibrated_cap(r['calls'][0]['seconds']) if r['calls'][0]['valid'] else 300)
    phases = ['initial','proposal','confirmation'] if a['admitted'] else ['initial']
    assert [c['phase'] for c in r['calls']]==phases
    losses, arrays = {}, {}
    for call in r['calls']:
        phase=call['phase']; p=stage/'cells'/phase
        v=json.loads(p.with_suffix('.json').read_text())
        assert v['valid']==call['valid'] and v['seconds']==call['seconds'] and v['cap']==call['cap']
        assert v['cap']==(300 if phase=='initial' else a['cap'])
        assert v['npz_sha256']==sha(p.with_suffix('.npz'))
        expected=initial if phase=='initial' else proposal
        assert v['graph_before']==v['graph_after']==expected
        if not v['valid']:
            continue
        folds=[]; arrays[phase]={}
        assert len(v['folds'])==3
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
            assert np.array_equal(z['classes'],np.unique(y))
            for i,(train,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SPLIT_SEED).split(y,y)):
                assert np.array_equal(z[f'rows_{i}'],valid) and np.array_equal(z[f'train_rows_{i}'],train)
                assert np.array_equal(z[f'truth_{i}'],y[valid])
                probs=z[f'probs_{i}'];folds.append(probability_loss(probs,y[valid],z['classes']))
                arrays[phase][i]=probs.copy()
                fitted=v['folds'][i]['fitted'];assert len(fitted)==1 and fitted[0]['categories']==cats
                assert v['folds'][i]['fold']==i
                assert 1<=fitted[0]['trees']<=trees
                if phase!='initial':
                    assert fitted[0]['trees']==trees
                    sig=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
                    pool=json.loads((stage/'pool-audits'/phase/'pools'/(sig+'.json')).read_text())
                    assert len(pool['pools'])==1 and pool['pools'][0]['rows']==train.tolist()
                    assert pool['pools'][0]['count']==len(train) and pool['pools'][0]['categories']==cats
                    assert pool['pools'][0]['columns']==(4 if smoke else 800)
                    assert pool['trees']==trees and pool['callback_count']==1
                    actual=pool['model_params'];excluded={'n_jobs','use_eval_set','enable_categorical','callbacks','fit_time_limit','early_stopping_rounds','od_wait','od_type','od_pval'}
                    params={k:v for k,v in expected[cat_index]['params'].items() if k not in excluded}
                    params.update(use_best_model=False,thread_count=8)
                    assert all(actual.get(k)==v for k,v in params.items())
                    assert not set(actual)&{'early_stopping_rounds','od_wait','od_type','od_pval'}
        assert abs(float(np.mean(folds))-v['fitness'])<1e-10
        losses[phase]=dict(cv=float(np.mean(folds)),folds=folds,seconds=v['seconds'])
    allvalid=len(losses)==3
    selected=allvalid and losses['confirmation']['cv']<=.99*losses['initial']['cv']
    assert r['returned_graph']==(proposal if selected else initial)
    if allvalid:
        assert all(np.array_equal(arrays['proposal'][i],arrays['confirmation'][i]) for i in range(3))
        assert r['evaluations_count']==1 and r['was_tuned']
        assert len(r['history'])==3 and all(len(g)==1 for g in r['history'])
        assert [g[0]['graph'] for g in r['history']]==[initial,proposal,r['returned_graph']]
        assert labels==['tuning_start','tuning_iteration_0','tuning_results']
        assert abs(r['init_metric']-losses['initial']['cv'])<1e-10
        assert abs(r['history'][0][0]['fitness']-losses['initial']['cv'])<1e-10
        assert abs(r['history'][1][0]['fitness']-losses['proposal']['cv'])<1e-10
        assert abs(r['history'][-1][0]['fitness']-(losses['confirmation' if selected else 'initial']['cv']))<1e-10
        assert r['obtained_metric']==r['history'][-1][0]['fitness']
    foldguard=allvalid and all(c<=1.02*b for c,b in zip(losses['confirmation']['folds'],losses['initial']['folds']))
    return dict(gate_passed=bool(selected and foldguard),integration_valid=allvalid,
                cells=losses,selected_allrow=bool(selected),policy_seconds=r['policy_seconds'],
                calibrated_cap=a['cap'],history_generations=len(r['history']),
                scope='Actual TunerBuilder + opt-in SimultaneousTuner subclass; split42 explored TRAIN; no composer/default/outer promotion')


def suite(stage):
    import fcntl
    from examples.advanced.diagnostics.fresh_budget_probe import training_conflict
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own=str(Path(__file__).resolve())
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:args=[v.decode(errors='replace') for v in p.read_bytes().split(b'\0') if v]
            except OSError:continue
            pid=int(p.parent.name)
            parent=pid==os.getppid() and args and Path(args[0]).name=='timeout' and own in args
            extra=any(Path(v).name in ['calibrated_tuner_probe.py','transport_fidelity_probe.py','api_budget_probe.py','allrow_fit_probe.py'] for v in args)
            if pid!=os.getpid() and not parent and (extra or training_conflict(pid,args,os.getpid(),os.getppid(),own)):
                raise RuntimeError('Host occupied: '+str(pid))
        manifest=json.loads((stage/'manifest.json').read_text());assert manifest['preflight_passed']
        for n,h in manifest['files'].items():assert sha(stage/n)==h,n
        write_json(stage/'STARTED',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
        before=time.monotonic()
        try:
            with (stage/'worker.log').open('x') as log:
                p=subprocess.Popen([sys.executable,own,'worker','--stage',str(stage)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=1100)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    code=p.returncode
            elapsed=time.monotonic()-before
            write_json(stage/'process.json',dict(exit_code=code,seconds=elapsed,wait_limit=1100,kill_grace=5))
            assert code==0 and elapsed<=1100
            result=audit(stage);result['stage_seconds']=time.monotonic()-before;assert result['stage_seconds']<=1140
            write_json(stage/'independent-audit.json',result)
            write_json(stage/'COMPLETE',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except BaseException as e:
            write_json(stage/'FAILED',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['suite','worker','audit']);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='suite':suite(a.stage)
    elif a.mode=='worker':run_tuner(a.stage)
    else:print(json.dumps(audit(a.stage)))
