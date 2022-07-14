from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.objective import Objective
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class ElitismTypesEnum(Enum):
    keep_n_best = 'keep_n_best'
    replace_worst = 'replace_worst'
    none = 'none'


class Elitism:
    def __init__(self, elitism_type: ElitismTypesEnum,
                 requirements: PipelineComposerRequirements,
                 objective: Objective):
        self.elitism_type = elitism_type
        self.requirements = requirements
        self.is_multi_objective = objective.is_multi_objective
        self._min_population_size_with_elitism = 5

    def __call__(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        if self.elitism_type is ElitismTypesEnum.none or not self.is_elitism_applicable():
            return new_population
        elif self.elitism_type is ElitismTypesEnum.keep_n_best:
            return self.keep_n_best_elitism(best_individuals, new_population)
        elif self.elitism_type is ElitismTypesEnum.replace_worst:
            return self.replace_worst_elitism(best_individuals, new_population)
        else:
            raise ValueError(f'Required elitism type not found: {self.elitism_type}')

    def keep_n_best_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        new_population[:len(best_individuals)] = best_individuals
        return new_population

    def replace_worst_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        population = best_individuals + new_population
        # sort population based on fitness value in ascending order
        sorted_population = sorted(population, key=lambda individual: individual.fitness)
        return sorted_population[:len(new_population)]

    def is_elitism_applicable(self):
        if self.is_multi_objective:
            return False
        else:
            return self.requirements.pop_size >= self._min_population_size_with_elitism
