from typing import Optional, Tuple

from deap.tools import ParetoFront as DeapParetoFront

from fedot.core.visualisation.opt_viz import PipelineEvolutionVisualiser


class ParetoFront(DeapParetoFront):
    def __init__(self, objective_names: Optional[Tuple[str]] = None, *args, **kwargs):
        self.objective_names = objective_names
        super(ParetoFront, self).__init__(*args, **kwargs)

    def show(self):
        PipelineEvolutionVisualiser().visualise_pareto(archive=self.items, show=True,
                                                       objectives_numbers=(1, 0),
                                                       objectives_names=self.objective_names[::-1])
