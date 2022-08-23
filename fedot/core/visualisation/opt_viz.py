from __future__ import annotations

import inspect
import os
from enum import Enum
from functools import partial
from typing import Optional, Union

from fedot.core.log import default_log
from fedot.core.visualisation.opt_history.fitness_box import visualise_fitness_box
from fedot.core.visualisation.opt_history.fitness_line import visualize_fitness_line, visualize_fitness_line_interactive
from fedot.core.visualisation.opt_history.operations_animated_bar import visualize_operations_animated_bar
from fedot.core.visualisation.opt_history.operations_kde import visualize_operations_kde


class PlotTypesEnum(Enum):
    fitness_line = partial(visualize_fitness_line)
    fitness_line_interactive = partial(visualize_fitness_line_interactive)
    fitness_box = partial(visualise_fitness_box)
    operations_kde = partial(visualize_operations_kde)
    operations_animated_bar = partial(visualize_operations_animated_bar)

    @classmethod
    def member_names(cls):
        return cls._member_names_


class OptHistoryVisualizer:
    def __init__(self, history):
        self.history = history
        for function in PlotTypesEnum:
            self.__setattr__(function.name, partial(function.value, history=self.history))

        self.log = default_log(self)

    def __call__(self, plot_type: Union[PlotTypesEnum, str] = PlotTypesEnum.fitness_box, **kwargs):
        """ Visualizes fitness values or operations used across generations.

        :param plot_type: visualization to show. Expected values are listed in
            'fedot.core.visualisation.opt_viz.PlotTypesEnum'.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed. Essential for animations.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of individuals with the best fitness per generation. The value should be in the
            interval (0, 1]. The other individuals are filtered out. The fraction will also be mentioned on the plot.
        :param show_fitness: if False, visualizations that support this argument will not display fitness.
        :param per_time: Shows time axis instead of generations axis. Currently, supported for plot types:
            'show_fitness_line', 'show_fitness_line_interactive'.
        :param use_tags: if True (default), all operations in the history are colored and grouped based on FEDOT
            repo tags. If False, operations are not grouped, colors are picked by fixed colormap for every history
            independently.
        """

        def check_args_constraints():
            nonlocal kwargs
            kwargs_to_ignore = []
            for argument in kwargs.keys():
                if argument not in visualization_parameters:
                    self.log.warning(f'Argument `{argument}` is not supported for "{plot_type.name}". It is ignored.')
                    kwargs_to_ignore.append(argument)
            kwargs = {key: value for key, value in kwargs.items() if key not in kwargs_to_ignore}
            # Check supported cases for `best_fraction`.
            best_fraction = kwargs.get('best_fraction')
            if (best_fraction is not None and
                    (best_fraction <= 0 or best_fraction > 1)):
                raise ValueError('`best_fraction` argument should be in the interval (0, 1].')
            # Check plot_type-specific cases
            per_time = kwargs.get('per_time')
            if (plot_type in (PlotTypesEnum.fitness_line, PlotTypesEnum.fitness_line_interactive) and
                    per_time and self.history.individuals[0][0].metadata.get('evaluation_time_iso') is None):
                self.log.warning('Evaluation time not found in optimization history. '
                                 'Showing fitness plot per generations...')
                kwargs['per_time'] = False
            elif plot_type is PlotTypesEnum.operations_animated_bar:
                save_path = kwargs.get('save_path')
                if not save_path:
                    raise ValueError('Argument `save_path` is required to save the animation.')

        if isinstance(plot_type, str):
            try:
                plot_type = PlotTypesEnum[plot_type]
            except KeyError:
                raise NotImplementedError(
                    f'Visualization "{plot_type}" is not supported. Expected values: '
                    f'{", ".join(PlotTypesEnum.member_names())}.')

        signature = inspect.signature(plot_type.value)
        visualization_parameters = signature.parameters
        default_kwargs = {p_name: p.default for p_name, p in visualization_parameters.items()
                          if p.default is not p.empty}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        check_args_constraints()

        self.log.info(
            'Visualizing optimization history... It may take some time, depending on the history size.')

        plot_type.value(self.history, **kwargs)
