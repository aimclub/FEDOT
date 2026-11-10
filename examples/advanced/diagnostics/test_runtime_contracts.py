"""Explicit, no-training regression checks for NEW isolated runtime copies.

Run with FEDOT_DIAGNOSTIC_VENDOR, or FEDOT_DIAGNOSTIC_HARNESS plus
FEDOT_DIAGNOSTIC_SOURCE/FEDOT_DIAGNOSTIC_PYTHON. Never patches imported code.
"""
import importlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def evaluation(individual, seconds, valid=True):
    from golem.core.optimisers.fitness.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
    return GraphEvalResult(individual.uid, SingleObjFitness(0.5 if valid else None),
                           individual.graph, dict(computation_time_in_seconds=seconds,
                                                  evaluation_time_iso=str(seconds)))


@pytest.fixture
def metadata_types():
    expected = os.environ.get('FEDOT_DIAGNOSTIC_VENDOR')
    if not expected:
        pytest.skip('requires an explicitly selected isolated vendor')
    import golem
    assert Path(golem.__file__).resolve().is_relative_to(Path(expected).resolve())
    from golem.core.optimisers.graph import OptGraph, OptNode
    from golem.core.optimisers.opt_history_objects.individual import Individual
    return Individual, lambda: OptGraph(OptNode('logit'))


def test_initial_template_and_evaluation_dict_are_not_modified(metadata_types):
    Individual, graph = metadata_types
    template = dict(use_input_preprocessing=True)
    a, b = [Individual(graph(), metadata=template) for _ in range(2)]
    result = evaluation(a, 10)
    a.set_evaluation_result(result)
    b.set_evaluation_result(evaluation(b, 20))
    assert template == dict(use_input_preprocessing=True)
    assert [x.metadata['computation_time_in_seconds'] for x in (a, b)] == [10, 20]
    assert a.metadata is not b.metadata and a.metadata is not result.metadata
    result.metadata['computation_time_in_seconds'] = 999
    assert a.metadata['computation_time_in_seconds'] == 10


@pytest.mark.parametrize('operator', ['mutation', 'crossover'])
def test_offspring_do_not_overwrite_parent_or_siblings(metadata_types, operator):
    from golem.core.optimisers.genetic.operators.crossover import Crossover
    from golem.core.optimisers.genetic.operators.mutation import Mutation
    Individual, graph = metadata_types
    template = dict(use_input_preprocessing=True)
    parent = Individual(graph(), metadata=template)
    parent.set_evaluation_result(evaluation(parent, 10))
    requirements = SimpleNamespace(static_individual_metadata=template)
    if operator == 'crossover':
        children = Crossover._get_individuals(
            SimpleNamespace(requirements=requirements), (graph(), graph()), (parent, parent), operator)
    else:
        def mutation_marker():
            pass
        mutation = SimpleNamespace(
            requirements=requirements,
            parameters=SimpleNamespace(max_num_of_operator_attempts=1),
            graph_generation_params=SimpleNamespace(verifier=lambda _: True),
            _operator_agent=SimpleNamespace(choose_action=lambda _: mutation_marker),
            _will_mutation_be_applied=lambda _: True,
            _apply_mutations=lambda *_: graph())
        children = [Mutation._mutation(mutation, parent)[0] for _ in range(2)]
    for child, seconds in zip(children, [20, 30]):
        child.set_evaluation_result(evaluation(child, seconds))
    assert [x.metadata['computation_time_in_seconds'] for x in [parent, *children]] == [10, 20, 30]
    assert template == dict(use_input_preprocessing=True)


def test_spawned_evaluations_applied_sequentially_keep_provenance(metadata_types):
    Individual, graph = metadata_types
    template = {'use_input_preprocessing': True}
    population = [Individual(graph(), metadata=template) for _ in range(2)]
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(evaluation, population, [10, 20]))
    for individual, result in zip(population, results):
        individual.set_evaluation_result(result)
    assert [x.metadata['computation_time_in_seconds'] for x in population] == [10, 20]
    assert template == {'use_input_preprocessing': True}


def test_history_serialization_keeps_individual_timings(metadata_types, tmp_path):
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    Individual, graph = metadata_types
    template = {'use_input_preprocessing': True}
    population = [Individual(graph(), metadata=template) for _ in range(2)]
    history = OptHistory(default_save_dir=tmp_path)
    for individual, seconds in zip(population, [10, 20]):
        individual.set_evaluation_result(evaluation(individual, seconds))
    history.add_to_history(population)
    loaded = OptHistory.load(history.save())
    assert [x.metadata['computation_time_in_seconds'] for x in loaded.generations[0]] == [10, 20]


def test_invalid_result_and_retry_do_not_pollute_template(metadata_types):
    Individual, graph = metadata_types
    template = {'use_input_preprocessing': True}
    individual = Individual(graph(), metadata=template)
    individual.set_evaluation_result(evaluation(individual, 10, valid=False))
    individual.set_evaluation_result(evaluation(individual, 20))
    assert individual.metadata['computation_time_in_seconds'] == 20
    assert template == {'use_input_preprocessing': True}
    with pytest.raises(ValueError):
        individual.set_evaluation_result(evaluation(individual, 30))
    assert individual.metadata['computation_time_in_seconds'] == 20


def test_fitness_only_update_preserves_existing_contract(metadata_types):
    from golem.core.optimisers.fitness.fitness import SingleObjFitness
    Individual, graph = metadata_types
    template = {'use_input_preprocessing': True}
    individual = Individual(graph(), metadata=template)
    new_graph = graph()
    individual.set_evaluation_result(SingleObjFitness(0.5), new_graph)
    assert individual.graph is new_graph and individual.fitness.value == 0.5
    assert individual.metadata == template


@pytest.mark.parametrize('enabled', [False, True])
def test_real_sanitized_amlb_boundary(enabled, monkeypatch, tmp_path):
    harness = os.environ.get('FEDOT_DIAGNOSTIC_HARNESS')
    if not harness:
        pytest.skip('requires an explicitly selected isolated AMLB harness')
    harness = Path(harness).resolve()
    source = Path(os.environ['FEDOT_DIAGNOSTIC_SOURCE']).resolve()
    from amlb.data import DatasetType
    from amlb.utils import Namespace
    from frameworks.shared import caller
    framework = importlib.import_module('frameworks.FEDOT')
    assert Path(framework.__file__).resolve().is_relative_to(harness)
    assert Path(caller.__file__).resolve().is_relative_to(harness)
    profile = source / 'examples/advanced/lgbm_400_defaults.json'
    key = 'FEDOT_OPERATION_DEFAULTS_OVERLAY'
    if enabled:
        monkeypatch.setenv(key, str(profile))
    else:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setenv('FEDOT_DIAGNOSTIC_UNRETAINED', 'must-not-leak')
    monkeypatch.setattr(caller, 'rconfig', lambda: Namespace(
        root_dir=str(harness), monitoring=Namespace(statistics=[])))
    real_launcher = caller.run_in_venv

    def launch_probe(caller_file, script_file, **kwargs):
        assert script_file == 'exec.py'
        # Only substitute the training script and interpreter. The real caller
        # still serializes input, sanitizes env, starts and reads a real child.
        return real_launcher(
            caller_file, str(Path(__file__).with_name('amlb_overlay_child_probe.py').resolve()),
            python_exec=os.environ['FEDOT_DIAGNOSTIC_PYTHON'], **kwargs)

    monkeypatch.setattr(caller, 'run_in_venv', launch_probe)
    split = SimpleNamespace(X='no-training-data', y_enc=0)
    dataset = SimpleNamespace(type=DatasetType.multiclass, target=None,
                              train=split, test=split, release=lambda: None)
    result = framework.run_fedot_tabular(dataset, Namespace(name='transport-probe'))
    probe = result['probe']
    assert Path(probe.fedot_file).is_relative_to(source)
    assert probe.overlay == (str(profile) if enabled else None)
    assert probe.constructor_rounds == (400 if enabled else 100)
    assert probe.explicit_rounds == 23
    assert probe.defaults['n_estimators'] == 400 if enabled else 'n_estimators' not in probe.defaults
    assert 'n_estimators' not in probe.regression_defaults
    assert probe.tmpdir == str(tmp_path) and probe.xdg_cache == str(tmp_path / 'cache')
    assert probe.unretained is None
