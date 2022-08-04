from random import shuffle

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT, Operator
from fedot.core.utilities.data_structures import ComparableEnum as Enum


class ElitismTypesEnum(Enum):
    keep_n_best = 'keep_n_best'
    replace_worst = 'replace_worst'
    none = 'none'


class Elitism(Operator):
    def __init__(self, elitism_type: ElitismTypesEnum,
                 requirements: PipelineComposerRequirements,
                 is_multi_objective: bool,
                 min_population_size_with_elitism: int = 5):
        self.elitism_type = elitism_type
        self.requirements = requirements
        self.is_multi_objective = is_multi_objective
        self.min_population_size_with_elitism = min_population_size_with_elitism

    def __call__(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        if self.elitism_type is ElitismTypesEnum.none or not self._is_elitism_applicable():
            return new_population
        elif self.elitism_type is ElitismTypesEnum.keep_n_best:
            return self._keep_n_best_elitism(best_individuals, new_population)
        elif self.elitism_type is ElitismTypesEnum.replace_worst:
            return self._replace_worst_elitism(best_individuals, new_population)
        else:
            raise ValueError(f'Required elitism type not found: {self.elitism_type}')

    def update_requirements(self, new_requirements: PipelineComposerRequirements):
        self.requirements = new_requirements

    def _is_elitism_applicable(self) -> bool:
        if self.is_multi_objective:
            return False
        return self.requirements.pop_size >= self.min_population_size_with_elitism

    def _keep_n_best_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        shuffle(new_population)
        new_population[:len(best_individuals)] = best_individuals
        return new_population

    def _replace_worst_elitism(self, best_individuals: PopulationT, new_population: PopulationT) -> PopulationT:
        population = best_individuals + new_population
        sorted_ascending_population = sorted(population, key=lambda individual: individual.fitness)
        return sorted_ascending_population[:len(new_population)]
