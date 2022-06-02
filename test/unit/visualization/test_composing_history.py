from pathlib import Path

import pytest

from fedot.core.optimisers.fitness.fitness import SingleObjFitness
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.utils import DEFAULT_PARAMS_STUB
from fedot.core.visualisation.opt_viz import PlotTypesEnum


def create_individual():
    first = OptNode(content={'name': 'logit', 'params': DEFAULT_PARAMS_STUB})
    second = OptNode(content={'name': 'lda', 'params': DEFAULT_PARAMS_STUB})
    final = OptNode(content={'name': 'knn', 'params': DEFAULT_PARAMS_STUB},
                    nodes_from=[first, second])

    indiviual = Individual(graph=OptGraph(final))
    indiviual.fitness = SingleObjFitness(1)
    return indiviual


def generate_history(generations_quantity, pop_size):
    history = OptHistory()
    for _ in range(generations_quantity):
        new_pop = []
        for _ in range(pop_size):
            ind = create_individual()
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


def test_history_adding():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)

    assert len(history.individuals) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.individuals[gen]) == pop_size


def test_individual_graph_type_is_optgraph():
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    for gen in range(generations_quantity):
        for ind in range(pop_size):
            assert type(history.individuals[gen][ind].graph) == OptGraph


def test_prepare_for_visualisation(capsys):
    generations_quantity = 2
    pop_size = 10
    history = generate_history(generations_quantity, pop_size)
    assert len(history.historical_pipelines) == pop_size * generations_quantity
    assert len(history.all_historical_fitness) == pop_size * generations_quantity

    history.print_leaderboard()
    captured = capsys.readouterr()
    assert 'n_lda_default_params' in captured.out
    assert 'Position' in captured.out

    dumped_history = history.save()
    loaded_history = OptHistory.load(dumped_history)
    loaded_history.print_leaderboard()
    captured = capsys.readouterr()
    assert 'n_lda_default_params' in captured.out
    assert 'Position' in captured.out


def test_all_historical_quality():
    pop_size = 4
    generations_quantity = 3
    history = generate_history(generations_quantity, pop_size)
    eval_fitness = [[0.9, 0.8], [0.8, 0.6], [0.2, 0.4], [0.9, 0.9]]
    weights = (-1, 1)
    for pop_num, population in enumerate(history.individuals):
        if pop_num != 0:
            eval_fitness = [[fit[0] + 0.5, fit[1]] for fit in eval_fitness]
        for pipeline_num, pipeline in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[pipeline_num], weights=weights)
            pipeline.fitness = fitness
    all_quality = history.all_historical_quality
    assert all_quality[0] == -0.9 and all_quality[4] == -1.4 and all_quality[5] == -1.3 and all_quality[10] == -1.2


@pytest.mark.parametrize('plot_type', PlotTypesEnum)
def test_history_show_saving_plots(tmp_path, plot_type: PlotTypesEnum):
    generations_quantity = 2
    pop_size = 5
    save_path = Path(tmp_path, plot_type.name)
    save_path = save_path.with_suffix('.gif') if plot_type is PlotTypesEnum.operations_animated_bar \
        else save_path.with_suffix('.png')
    history = generate_history(generations_quantity, pop_size)
    history.show(plot_type=plot_type, save_path=str(save_path), pct_best=0.1)
    assert save_path.exists()
