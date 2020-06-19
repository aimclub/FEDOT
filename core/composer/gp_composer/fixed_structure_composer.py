from dataclasses import dataclass
from typing import (
    Callable,
    Optional,
)

from core.composer.chain import Chain
from core.composer.composer import ComposerRequirements
from core.composer.gp_composer.gp_composer import GPComposer
from core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from core.composer.optimisers.mutation import MutationTypesEnum
from core.models.data import InputData


@dataclass
class GPComposerRequirements(ComposerRequirements):
    pop_size: Optional[int] = 50
    num_of_generations: Optional[int] = 50
    crossover_prob: Optional[float] = None
    mutation_prob: Optional[float] = None


class FixedStructureComposer(GPComposer):
    def __init__(self):
        super().__init__()

    def compose_chain(self, data: InputData, initial_chain: Chain,
                      composer_requirements: Optional[GPComposerRequirements],
                      metrics: Optional[Callable],
                      optimiser_parameters: GPChainOptimiserParameters = None,
                      is_visualise: bool = False) -> Chain:
        composer_requirements.crossover_prob = 0.0

        fixed_structure_optimiser_parameters = GPChainOptimiserParameters(
            mutation_types=[MutationTypesEnum.simple])

        return super().compose_chain(data, initial_chain,
                                     composer_requirements,
                                     metrics,
                                     fixed_structure_optimiser_parameters,
                                     is_visualise)
