from typing import (
    List,
    Callable,
    Optional
)
from core.models.model import Model
from core.composer.chain import Chain


class GPChainOptimiser():
    def __init__(self, initial_chains, requirements, input_data):
        if not initial_chains:
            self.population = [Chain()._flat_nodes_tree(requirements,input_data ) for i in range(requirements.pop_size)]
        else:
            self.population = initial_chains


    def run_evolution(self) -> Chain:
        return Chain()

    '''
    def run_evolution(self):
        average_num_of_generation = None

        for i in range(self.max_generations):
            print("generation num:\n", i)
            selected_indexes = self.selection(self.population, self.minimization)
            new_population = []
            for ind_num in range(self.population_size - 1):
                new_population.append(self.crossover(self.population[selected_indexes[ind_num][0]],
                                                     self.population[selected_indexes[ind_num][1]], ind_num, i))
                new_population[ind_num] = self.mutation(new_population[ind_num])
                new_population[ind_num].set_fitness()

            self.population = deepcopy(new_population)
            self.population.append(self.the_best_ind)

            all_fit = [ind.fitness for ind in self.population]
            if self.minimization:
                self.the_best_ind = self.population[np.argmin(all_fit)]
            else:
                self.the_best_ind = self.population[np.argmax(all_fit)]

        return self.the_best_ind.get_fitness(sample=self.problem.get_test_set())
    '''