from enum import Enum
from typing import Union

from fedot.core.log import default_log
from fedot.core.visualisation.opt_history.fitness_box import FitnessBox
from fedot.core.visualisation.opt_history.fitness_line import FitnessLine, FitnessLineInteractive
from fedot.core.visualisation.opt_history.operations_animated_bar import OperationsAnimatedBar
from fedot.core.visualisation.opt_history.operations_kde import OperationsKDE


class PlotTypesEnum(Enum):
    fitness_box = FitnessBox
    fitness_line = FitnessLine
    fitness_line_interactive = FitnessLineInteractive
    operations_kde = OperationsKDE
    operations_animated_bar = OperationsAnimatedBar

    @classmethod
    def member_names(cls):
        return cls._member_names_


class OptHistoryVisualizer:
    def __init__(self, history):
        self.history = history
        self.fitness_box = FitnessBox(self.history).visualize
        self.fitness_line = FitnessLine(self.history).visualize
        self.fitness_line_interactive = FitnessLineInteractive(self.history).visualize
        self.operations_kde = OperationsKDE(self.history).visualize
        self.operations_animated_bar = OperationsAnimatedBar(self.history).visualize

        self.log = default_log(self)

    def __call__(self, plot_type: Union[PlotTypesEnum, str] = PlotTypesEnum.fitness_box, **kwargs):
        """ Visualizes fitness values or operations used across generations.

        :param plot_type: visualization to show. Expected values are listed in
            'fedot.core.visualisation.opt_viz.PlotTypesEnum'.
        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
        it will be displayed. Essential for animations.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the
            visualization. Must be in the interval (0, 1].
        :param show_fitness: if False, visualizations that support this argument will not display fitness.
        :param per_time: Shows time axis instead of generations axis. Currently, supported for plot types:
            'show_fitness_line', 'show_fitness_line_interactive'.
        :param use_tags: if True (default), all operations in the history are colored and grouped based on
            FEDOT repo tags. If False, operations are not grouped, colors are picked by fixed colormap for
            every history independently.
        """

        if isinstance(plot_type, str):
            try:
                visualize_function = vars(self)[plot_type]
            except KeyError:
                raise NotImplementedError(
                    f'Visualization "{plot_type}" is not supported. Expected values: '
                    f'{", ".join(PlotTypesEnum.member_names())}.')
        else:
            visualize_function = vars(self)[plot_type.name]
        visualize_function(**kwargs)
