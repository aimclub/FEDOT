from typing import Tuple

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.gp_comp.parameters.mutation_prob import AdaptiveMutationProb
from fedot.core.optimisers.gp_comp.parameters.parameter import AdaptiveParameter, ConstParameter

VariationOperatorProb = AdaptiveParameter[Tuple[float, float]]


class AdaptiveVariationProb(VariationOperatorProb):

    def __init__(self, mutation_prob: float = 0.5, crossover_prob: float = 0.5):
        self._mutation_prob_param = AdaptiveMutationProb(mutation_prob)
        self._mutation_prob = self._mutation_prob_param.initial
        self._crossover_prob_init = crossover_prob
        self._crossover_prob = crossover_prob

    @property
    def mutation_prob(self):
        return self._mutation_prob

    @property
    def crossover_prob(self):
        return self._crossover_prob

    @property
    def initial(self) -> Tuple[float, float]:
        return self._mutation_prob_param.initial, self._crossover_prob_init

    def next(self, population: PopulationT) -> Tuple[float, float]:
        self._mutation_prob = self._mutation_prob_param.next(population)
        self._crossover_prob = 1. - self._mutation_prob
        return self._mutation_prob, self._crossover_prob


def init_adaptive_operators_prob(genetic_scheme_type: GeneticSchemeTypesEnum,
                                 requirements: PipelineComposerRequirements) -> VariationOperatorProb:
    if genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
        operators_prob = AdaptiveVariationProb()
    else:
        operators_prob = ConstParameter((requirements.mutation_prob, requirements.crossover_prob))
    return operators_prob

