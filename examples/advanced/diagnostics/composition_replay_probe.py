"""H20: no-fit replay of historical selected graphs and elapsed composition time.

External run STARTED is earlier than API fit. Its elapsed time gives a conservative
remaining-budget bound, not a reconstruction of the original monotonic fit clock.
Only TRAIN history, timing lines and audited H19 costs are read. No dataset loads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def elapsed_bound(started, tuning_line):
    start = datetime.fromisoformat(started.strip().replace('Z', '+00:00'))
    matches = re.findall(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - ApiComposer - Hyperparameters tuning started', tuning_line)
    assert len(matches) == 1, 'Unique tuning entry required'
    end = datetime.strptime(matches[0], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc)
    seconds = (end-start).total_seconds()
    assert 0 < seconds < 3240, 'Historical chronology outside API envelope'
    return seconds


def selected_graph(history_path, expected_uid):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    history = OptHistory.load(str(history_path))
    choices = [g for g in history.generations if g.label == 'final_choices']
    assert len(choices) == 1 and len(choices[0]) == 1
    individual = choices[0][0]
    assert str(individual.uid) == expected_uid and individual.fitness.valid
    return PipelineAdapter().restore(individual.graph)


def replay_dispatch(stage, pipeline, elapsed):
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from examples.advanced.diagnostics import composer_paid_policy as policy
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    from fedot.core.pipelines.tuning import tuner_builder
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    adapter = load_adapter(stage)
    config = SimpleNamespace(type='classification', seed=104735,
                             framework_params={'_paid_allrow_tuning': True})
    model = adapter._make_fedot(config, 'neg_log_loss', dict(n_jobs=8, cv_folds=3,
        with_tuning=True, use_operations_cache=False, use_preprocessing_cache=False,
        use_predictions_cache=False, final_fit_use_remaining_time=False), 54, 5.4)
    from fedot.api.time import ApiTime
    model.api_composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    model.params.init_params_for_composing(model.api_composer.timer.timedelta_composing,
                                          multi_objective=False)
    model._fit_started_at = 1000.
    counts = dict(builder_entries=0, fits=0, evaluations=0)

    class AtBuilder(Exception):
        pass

    def stop_builder(*args, **kwargs):
        counts['builder_entries'] += 1
        raise AtBuilder()

    def forbidden_fit(*args, **kwargs):
        counts['fits'] += 1
        raise AssertionError('H20 forbids model fits')

    def forbidden_eval(*args, **kwargs):
        counts['evaluations'] += 1
        raise AssertionError('H20 forbids objective evaluations')

    with patch.object(Pipeline, 'fit', forbidden_fit), \
         patch.object(PipelineObjectiveEvaluate, '__call__', forbidden_eval), \
         patch.object(tuner_builder, 'TunerBuilder', stop_builder), \
         patch.object(policy.time, 'monotonic', return_value=1000.+elapsed):
        try:
            model.api_composer.tune_final_pipeline(
                SimpleNamespace(task=Task(TaskTypesEnum.classification)), pipeline, OptHistory())
        except AtBuilder:
            pass
        else:
            raise AssertionError('Historical graph did not enter paid tuner builder')
    record = model.api_composer._experimental_paid_policy.record
    assert record['status'] == 'running' and record['calls'] == []
    assert counts == dict(builder_entries=1, fits=0, evaluations=0)
    return dict(bounds=record['bounds'], graph=record['initial_graph'], counts=counts,
                stopped_before_builder=True)


def run(stage):
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    reference = json.loads((stage/'h19-result.json').read_text())
    results = {}
    for side in ['baseline', 'candidate']:
        evidence = json.loads((stage/f'{side}-evidence.json').read_text())
        assert sha(stage/f'{side}-history.json') == evidence['history_sha256']
        pipe = selected_graph(stage/f'{side}-history.json', evidence['selected_uid'])
        assert graph_record(pipe) == reference['policy']['initial_graph']
        elapsed = elapsed_bound(evidence['started'], evidence['tuning_line'])
        results[side] = dict(elapsed_upper_bound=elapsed,
                            dispatch=replay_dispatch(stage, pipe, elapsed))
    save(stage/'result.json', results)


def audit(stage):
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    results = json.loads((stage/'result.json').read_text())
    h19 = json.loads((stage/'h19-result.json').read_text())
    measured = json.loads((stage/'h19-audit.json').read_text())
    assert measured['gate_passed']
    out = {}
    for side in ['baseline', 'candidate']:
        e = json.loads((stage/f'{side}-evidence.json').read_text())
        assert sha(stage/f'{side}-history.json') == e['history_sha256']
        graph = graph_record(selected_graph(stage/f'{side}-history.json', e['selected_uid']))
        elapsed = elapsed_bound(e['started'], e['tuning_line'])
        r = results[side];d = r['dispatch'];b = d['bounds']
        assert r['elapsed_upper_bound'] == elapsed
        assert d['graph'] == graph == h19['policy']['initial_graph']
        assert d['counts'] == dict(builder_entries=1, fits=0, evaluations=0)
        assert d['stopped_before_builder']
        remaining = 3240-elapsed
        allocation = min(1100, remaining-324-162)
        assert allocation >= 115
        assert abs(b['remaining']-remaining) < 1e-8
        assert b['allocation'] == allocation and b['refit_reserve'] == 324 and b['postfit_reserve'] == 162
        assert b['initial_cap'] == min(300, (allocation-100)/3)
        calls = h19['policy']['calls']
        cap = min(b['initial_cap'], max(5, 1.3*calls[0]['seconds']))
        # Entire measured H19 tuning overhead is charged before the replayed trial.
        overhead = h19['policy']['charged_tuning_seconds']-sum(c['seconds'] for c in calls)
        replayed_before_trial = allocation-calls[0]['seconds']-overhead
        replayed_tail = measured['tuning_seconds']+measured['refit_seconds']+162
        out[side] = dict(elapsed_upper_bound=elapsed, remaining_lower_bound=remaining,
            allocation=allocation, initial_cap=b['initial_cap'], calibrated_cap=cap,
            replayed_before_trial=replayed_before_trial, trial_reserve=2*(cap+50),
            measured_h19_tail_plus_postfit=replayed_tail,
            gate_passed=bool(b['initial_cap'] >= calls[0]['seconds'] and
                replayed_before_trial >= 2*(cap+50) and
                measured['tuning_seconds'] <= allocation and replayed_tail <= remaining))
    return dict(gate_passed=all(v['gate_passed'] for v in out.values()), cases=out,
                actual_fits=0, objective_evaluations=0,
                scope='Historical wall-clock conservative replay; no future runtime, evolution or outer-quality claim')


def suite(stage):
    import fcntl
    with (stage.parents[2]/'experiments/pr1467-followup-host.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (stage/'STARTED').exists()
        manifest = json.loads((stage/'manifest.json').read_text())
        assert manifest['preflight_passed']
        for name, digest in manifest['files'].items():
            assert sha(stage/name) == digest, name
        save(stage/'STARTED', dict(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
        before = time.monotonic()
        try:
            run(stage)
            result = audit(stage)
            result['stage_seconds'] = time.monotonic()-before
            assert result['stage_seconds'] <= 180
            save(stage/'independent-audit.json', result)
            save(stage/'COMPLETE', dict(utc=datetime.now(timezone.utc).isoformat()))
        except BaseException as exc:
            save(stage/'FAILED', dict(error=repr(exc)))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['suite', 'audit'])
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'suite':
        suite(args.stage)
    else:
        print(json.dumps(audit(args.stage)))
