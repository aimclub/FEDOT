"""Additional read-only H23 audit of original-tuner selection and history lineage."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(stage):
    from examples.advanced.diagnostics.cache_startup_contract import verify
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from fedot.core.pipelines.adapters import PipelineAdapter
    count=verify(stage);sides={};adapter=PipelineAdapter()
    for side in ['prepared-false','prepared-true']:
        d=stage/'runs'/side/'observation'
        read=lambda name:json.loads((d/name).read_text())
        entry=read('tuning-entry.json');exit=read('tuning-exit.json')
        calls=exit['evaluations'];returned=read('tuning-return.json')['graph']
        assert all(c['valid'] for c in calls)
        history=OptHistory.load(str(d/'history.json'))
        starts=[g for g in history.generations if g.label=='tuning_start']
        ends=[g for g in history.generations if g.label=='tuning_results']
        trials=[ind for g in history.generations if str(g.label).startswith('tuning_iteration_') for ind in g]
        assert len(starts)==len(ends)==1 and len(starts[0])==len(ends[0])==1
        initial=starts[0][0];final=ends[0][0]
        assert graph_record(adapter.restore(initial.graph))==calls[0]['graph']==entry['graph']
        assert initial.fitness.value==calls[0]['fitness']
        assert len(trials)==len(calls)-2
        for ind,call in zip(trials,calls[1:-1]):
            assert graph_record(adapter.restore(ind.graph))==call['graph']
            assert ind.fitness.valid and ind.fitness.value==call['fitness']
            assert [p.uid for p in ind.parent_operator.parent_individuals]==[initial.uid]
        assert [p.uid for p in final.parent_operator.parent_individuals]==[initial.uid]
        # Original BaseTuner default deviation is 0.05 percent, not H18's 1 percent.
        selected=calls[-1]['fitness']<=calls[0]['fitness']*(1-.05/100)
        expected=calls[-1] if selected else calls[0]
        assert returned==expected['graph']==graph_record(adapter.restore(final.graph))
        assert final.fitness.value==expected['fitness']
        assert any(c['graph']==calls[-1]['graph'] and c['fitness']==calls[-1]['fitness'] for c in calls[1:-1])
        assert exit['policy']==({'status':'ineligible_original_path'} if side.endswith('-true') else None)
        sides[side]=dict(valid_trials=len(trials),selected_tuned=selected,
            initial_cv=calls[0]['fitness'],confirmation_cv=calls[-1]['fitness'],
            returned_fitness=final.fitness.value,history_sha256=hashlib.sha256((d/'history.json').read_bytes()).hexdigest())
    return dict(passed=True,verified_files=count,sides=sides,
        scope='Original fallback tuner selection/lineage; synthetic TRAIN CV only, no policy quality comparison')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    print(json.dumps(audit(p.parse_args().stage),indent=2))
