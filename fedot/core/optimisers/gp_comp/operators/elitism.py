from random import shuffle

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class ElitismTypesEnum(Enum):
    keep_n_best = 'keep_n_best'
    replace_worst = 'replace_worst'
    none = 'none'


class Elitism(Operator):
    def __init__(self, parameters: 'GPGraphOptimizerParameters',
                 is_multi_objective: bool,
                 min_population_size_with_elitism: int = 5):  # TODO: move to requirements
        super().__init__(parameters=parameters)
        self.is_multi_objective = is_multi_objective
        self.min_population_size_with_elitism = min_population_size_with_elitism

    def __call__(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        elitism_type = self.parameters.elitism_type
        if elitism_type is ElitismTypesEnum.none or not self._is_elitism_applicable():
            return new_population
        elif elitism_type is ElitismTypesEnum.keep_n_best:
            return self._keep_n_best_elitism(best_individuals, new_population)
        elif elitism_type is ElitismTypesEnum.replace_worst:
            return self._replace_worst_elitism(best_individuals, new_population)
        else:
            raise ValueError(f'Required elitism type not found: {elitism_type}')

    def _is_elitism_applicable(self) -> bool:
        if self.is_multi_objective:
            return False
        return self.parameters.pop_size >= self.min_population_size_with_elitism

    def _keep_n_best_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        shuffle(new_population)
        new_population[:len(best_individuals)] = best_individuals
        return new_population

    def _replace_worst_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        population = best_individuals + new_population
        sorted_ascending_population = sorted(population, key=lambda individual: individual.fitness)
        return sorted_ascending_population[:len(new_population)]
