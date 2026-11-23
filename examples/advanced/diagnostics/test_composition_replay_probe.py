import os
from pathlib import Path
import pytest
from examples.advanced.diagnostics.composition_replay_probe import elapsed_bound, replay_dispatch


def test_conservative_elapsed_preserves_milliseconds_and_midnight():
    assert elapsed_bound('2026-11-22T23:59:00Z',
        '2026-11-23 00:01:00,123 - ApiComposer - Hyperparameters tuning started') == 120.123


@pytest.mark.parametrize('line', [
    '2026-11-22 12:00:00,000 - ApiComposer - Hyperparameters tuning started',
    '2026-11-22 13:00:00,000 - ApiComposer - Hyperparameters tuning started',
    'Composition time: 23 min',
    '2026-11-22 12:10:00,000 - ApiComposer - Hyperparameters tuning started\n'
    '2026-11-22 12:11:00,000 - ApiComposer - Hyperparameters tuning started'])
def test_reject_ambiguous_or_out_of_budget_chronology(line):
    with pytest.raises(AssertionError):
        elapsed_bound('2026-11-22T12:00:00Z', line)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'), reason='frozen host API composer')
def test_actual_dispatch_charges_prior_composition_without_any_fit(tmp_path):
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    (tmp_path/'harness').symlink_to(Path(os.environ['FEDOT_H9_ADAPTER']).parents[2], target_is_directory=True)
    pipe = Pipeline(PipelineNode('catboost', nodes_from=[PipelineNode('scaling')]))
    pipe.replace_n_jobs_in_nodes(8)
    result = replay_dispatch(tmp_path, pipe, 2000)
    assert result['bounds']['remaining'] == 1240
    assert result['bounds']['allocation'] == 754
    assert result['bounds']['initial_cap'] == 218
    assert result['counts'] == dict(builder_entries=1, fits=0, evaluations=0)


@pytest.mark.skipif(not os.environ.get('FEDOT_H9_ADAPTER'), reason='frozen history and pipeline adapter')
def test_serialized_selected_graph_identity(tmp_path):
    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.adapters import PipelineAdapter
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from golem.core.optimisers.fitness import SingleObjFitness
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record
    from examples.advanced.diagnostics.composition_replay_probe import selected_graph
    pipe = Pipeline(PipelineNode('catboost', nodes_from=[PipelineNode('scaling')]))
    pipe.replace_n_jobs_in_nodes(8)
    individual = Individual(PipelineAdapter().adapt(pipe), fitness=SingleObjFitness(.9))
    history = OptHistory()
    history.add_to_history([individual], generation_label='final_choices')
    path = tmp_path/'history.json'
    history.save(str(path))
    assert graph_record(selected_graph(path, str(individual.uid))) == graph_record(pipe)
    with pytest.raises(AssertionError):
        selected_graph(path, 'different-selected-uid')
