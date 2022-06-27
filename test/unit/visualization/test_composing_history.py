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

    individual = Individual(graph=OptGraph(final))
    individual.set_evaluation_result(SingleObjFitness(1))
    return individual


@pytest.fixture(scope='module')
def generate_history(request):
    generations_quantity, pop_size = request.param
    history = OptHistory()
    for gen_num in range(generations_quantity):
        new_pop = []
        for _ in range(pop_size):
            ind = create_individual()
            ind.set_native_generation(gen_num)
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


@pytest.mark.parametrize('generate_history', [[2, 10]], indirect=True)
def test_history_adding(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history

    assert len(history.individuals) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.individuals[gen]) == pop_size


@pytest.mark.parametrize('generate_history', [[2, 10]], indirect=True)
def test_individual_graph_type_is_optgraph(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    for gen in range(generations_quantity):
        for ind in range(pop_size):
            assert type(history.individuals[gen][ind].graph) == OptGraph


@pytest.mark.parametrize('generate_history', [[2, 10]], indirect=True)
def test_prepare_for_visualisation(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    assert len(history.historical_pipelines) == pop_size * generations_quantity
    assert len(history.all_historical_fitness) == pop_size * generations_quantity

    leaderboard = history.get_leaderboard()
    assert OptNode('lda').descriptive_id in leaderboard
    assert 'Position' in leaderboard

    dumped_history = history.save()
    loaded_history = OptHistory.load(dumped_history)
    leaderboard = loaded_history.get_leaderboard()
    assert OptNode('lda').descriptive_id in leaderboard
    assert 'Position' in leaderboard


@pytest.mark.parametrize('generate_history', [[3, 4]], indirect=True)
def test_all_historical_quality(generate_history):
    history = generate_history
    eval_fitness = [[0.9, 0.8], [0.8, 0.6], [0.2, 0.4], [0.9, 0.9]]
    weights = (-1, 1)
    for pop_num, population in enumerate(history.individuals):
        if pop_num != 0:
            eval_fitness = [[fit[0] + 0.5, fit[1]] for fit in eval_fitness]
        for ind_num, individual in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[ind_num], weights=weights)
            object.__setattr__(individual, 'fitness', fitness)
    all_quality = history.all_historical_quality
    assert all_quality[0] == -0.9 and all_quality[4] == -1.4 and all_quality[5] == -1.3 and all_quality[10] == -1.2


@pytest.mark.parametrize('generate_history', [[3, 4]], indirect=True)
@pytest.mark.parametrize('plot_type', PlotTypesEnum)
def test_history_show_saving_plots(tmp_path, plot_type: PlotTypesEnum, generate_history):
    save_path = Path(tmp_path, plot_type.name)
    save_path = save_path.with_suffix('.gif') if plot_type is PlotTypesEnum.operations_animated_bar \
        else save_path.with_suffix('.png')
    history = generate_history
    history.show(plot_type=plot_type, save_path=str(save_path), best_fraction=0.1, dpi=100)
    if plot_type is not PlotTypesEnum.fitness_line_interactive:
        assert save_path.exists()
