from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class ElitismTypesEnum(Enum):
    keep_n_best = 'keep_n_best'
    replace_worst = 'replace_worst'
    none = 'none'


class Elitism:
    def __init__(self, elitism_type: ElitismTypesEnum):
        self.elitism_type = elitism_type

    def __call__(self, best_individuals: PopulationT, new_population: PopulationT):
        if self.elitism_type is ElitismTypesEnum.keep_n_best:
            return self.keep_n_best_elitism(best_individuals, new_population)
        elif self.elitism_type is ElitismTypesEnum.replace_worst:
            return self.replace_worst_elitism(best_individuals, new_population)
        elif self.elitism_type is ElitismTypesEnum.none:
            return new_population
        else:
            raise ValueError(f'Required elitism type not found: {type}')

    def keep_n_best_elitism(self, best_individuals: PopulationT, new_population: PopulationT):
        new_population[:len(best_individuals)] = best_individuals
        return new_population

    def replace_worst_elitism(self, best_individuals: PopulationT, new_population: PopulationT):
        population = best_individuals + new_population
        # sort population based on fitness value in descending order
        sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        return sorted_population[:len(new_population)]
