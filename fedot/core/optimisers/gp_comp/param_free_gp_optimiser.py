from typing import Optional, Union, Sequence

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import Log
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.pipelines.pipeline import Pipeline


class EvoGraphParameterFreeOptimiser(EvoGraphOptimiser):
    """
    Implementation of the parameter-free adaptive evolutionary optimiser
    (population size and genetic operators rates is changing over time).
    For details, see https://ieeexplore.ieee.org/document/9504773
    """

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Pipeline, Sequence[Pipeline]],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 parameters: Optional[GPGraphOptimiserParameters] = None,
                 log: Log = None):
        super().__init__(objective, initial_graph, requirements, graph_generation_params, parameters, log)
