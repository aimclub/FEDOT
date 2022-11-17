import functools
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from fedot.core.log import default_log
from fedot.core.optimisers.fitness import null_fitness
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization
from fedot.core.visualisation.opt_history.utils import show_or_save_figure


def with_alternate_matplotlib_backend(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        default_mpl_backend = mpl.get_backend()
        try:
            mpl.use('TKAgg')
            return func(*args, **kwargs)
        except ImportError as e:
            default_log(prefix='Requirements').warning(e)
        finally:
            mpl.use(default_mpl_backend)

    return wrapper


def setup_fitness_plot(axis: plt.Axes, xlabel: str, title: Optional[str] = None, with_legend: bool = False):
    if axis is None:
        fig, axis = plt.subplots()

    if with_legend:
        axis.legend()
    axis.set_ylabel('Fitness')
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis='y')


def plot_fitness_line_per_time(axis: plt.Axes, generations: List[List[Individual]], label: Optional[str] = None,
                               with_generation_limits: bool = True) \
        -> Dict[int, Individual]:
    best_fitness = null_fitness()
    gen_start_times = []
    best_individuals = {}

    start_time = datetime.fromisoformat(
        min(generations[0], key=lambda ind: ind.metadata['evaluation_time_iso']).metadata[
            'evaluation_time_iso'])
    end_time_seconds = (datetime.fromisoformat(
        max(generations[-1], key=lambda ind: ind.metadata['evaluation_time_iso']).metadata[
            'evaluation_time_iso']) - start_time).seconds

    for gen_num, gen in enumerate(generations):
        gen_start_times.append(1e10)
        gen_sorted = sorted(gen, key=lambda ind: ind.metadata['evaluation_time_iso'])
        for ind in gen_sorted:
            if ind.native_generation != gen_num:
                continue
            evaluation_time = (datetime.fromisoformat(ind.metadata['evaluation_time_iso']) - start_time).seconds
            if evaluation_time < gen_start_times[gen_num]:
                gen_start_times[gen_num] = evaluation_time
            if ind.fitness > best_fitness:
                best_individuals[evaluation_time] = ind
                best_fitness = ind.fitness

    best_eval_times, best_fitnesses = np.transpose(
        [(evaluation_time, abs(individual.fitness.value))
         for evaluation_time, individual in best_individuals.items()])

    best_eval_times = list(best_eval_times)
    best_fitnesses = list(best_fitnesses)

    if best_eval_times[-1] != end_time_seconds:
        best_fitnesses.append(abs(best_fitness.value))
        best_eval_times.append(end_time_seconds)
    gen_start_times.append(end_time_seconds)

    axis.step(best_eval_times, best_fitnesses, where='post', label=label)

    if with_generation_limits:
        axis_gen = axis.twiny()
        axis_gen.set_xlim(axis.get_xlim())
        axis_gen.set_xticks(gen_start_times, list(range(len(gen_start_times) - 1)) + [''])
        axis_gen.locator_params(nbins=10)
        axis_gen.set_xlabel('Generation')

        gen_ticks = axis_gen.get_xticks()
        prev_time = gen_ticks[0]
        axis.axvline(prev_time, color='k', linestyle='--', alpha=0.3)
        for i, next_time in enumerate(gen_ticks[1:]):
            axis.axvline(next_time, color='k', linestyle='--', alpha=0.3)
            if i % 2 == 0:
                axis.axvspan(prev_time, next_time, color='k', alpha=0.05)
            prev_time = next_time

    return best_individuals


def plot_fitness_line_per_generations(axis: plt.Axes, generations, label: Optional[str] = None) \
        -> Dict[int, Individual]:
    best_fitness = null_fitness()
    best_individuals = {}

    for gen_num, gen in enumerate(generations):
        for ind in gen:
            if ind.native_generation != gen_num:
                continue
            if ind.fitness > best_fitness:
                best_individuals[gen_num] = ind
                best_fitness = ind.fitness

    best_generations, best_fitnesses = np.transpose(
        [(gen_num, abs(individual.fitness.value))
         for gen_num, individual in best_individuals.items()])

    best_generations = list(best_generations)
    best_fitnesses = list(best_fitnesses)

    if best_generations[-1] != len(generations) - 1:
        best_fitnesses.append(abs(best_fitness.value))
        best_generations.append(len(generations) - 1)

    axis.step(best_generations, best_fitnesses, where='post', label=label)
    axis.set_xticks(range(len(generations)))
    axis.locator_params(nbins=10)
    return best_individuals


class FitnessLine(HistoryVisualization):
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  per_time: Optional[bool] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param dpi: DPI of the output figure.
        :param per_time: defines whether to show time grid if it is available in history.
        """
        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        per_time = per_time if per_time is not None else self.get_predefined_value('per_time') or False

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
        if per_time:
            xlabel = 'Time, s'
            plot_fitness_line_per_time(ax, self.history.individuals)
        else:
            xlabel = 'Generation'
            plot_fitness_line_per_generations(ax, self.history.individuals)
        setup_fitness_plot(ax, xlabel)
        show_or_save_figure(fig, save_path, dpi)


class FitnessLineInteractive(HistoryVisualization):

    @with_alternate_matplotlib_backend
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  per_time: Optional[bool] = None,  graph_show_kwargs: Optional[Dict[str, Any]] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        Additionally, shows the structure of the best individuals and the moment of their discovering.
        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
            it will be displayed.
        :param dpi: DPI of the output figure.
        :param per_time: defines whether to show time grid if it is available in history.
        :param graph_show_kwargs: keyword arguments of `graph.show()` function.
        """

        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        per_time = per_time if per_time is not None else self.get_predefined_value('per_time') or False
        graph_show_kwargs = graph_show_kwargs or self.get_predefined_value('graph_show_params') or {}

        graph_show_kwargs = graph_show_kwargs or self.visualizer.visuals_params.get('graph_show_params') or {}

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        ax_fitness, ax_graph = axes

        if per_time:
            x_label = 'Time, s'
            x_template = 'time {} s'
            plot_func = plot_fitness_line_per_time
        else:
            x_label = 'Generation'
            x_template = 'generation {}'
            plot_func = plot_fitness_line_per_generations

        best_individuals = plot_func(ax_fitness, self.history.individuals)
        setup_fitness_plot(ax_fitness, x_label)

        ax_graph.axis('off')

        class InteractivePlot:
            temp_path = Path(default_fedot_data_dir(), 'current_graph.png')

            def __init__(self, best_individuals: Dict[int, Individual]):
                self.best_x: List[int] = list(best_individuals.keys())
                self.best_individuals: List[Individual] = list(best_individuals.values())
                self.index: int = len(self.best_individuals) - 1
                self.time_line = ax_fitness.axvline(self.best_x[self.index], color='r', alpha=0.7)
                self.graph_images: List[np.ndarray] = []
                self.generate_graph_images()
                self.update_graph()

            def generate_graph_images(self):
                for ind in self.best_individuals:
                    graph = ind.graph
                    graph.show(self.temp_path, **graph_show_kwargs)
                    self.graph_images.append(plt.imread(str(self.temp_path)))
                self.temp_path.unlink()

            def update_graph(self):
                ax_graph.imshow(self.graph_images[self.index])
                x = self.best_x[self.index]
                fitness = self.best_individuals[self.index].fitness
                ax_graph.set_title(f'The best pipeline at {x_template.format(x)}, fitness={fitness}')

            def update_time_line(self):
                self.time_line.set_xdata(self.best_x[self.index])

            def step_index(self, step: int):
                self.index = (self.index + step) % len(self.best_individuals)
                self.update_graph()
                self.update_time_line()
                plt.draw()

            def next(self, event):
                self.step_index(1)

            def prev(self, event):
                self.step_index(-1)

        callback = InteractivePlot(best_individuals)

        if not save_path:  # display buttons only for an interactive plot
            ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
            b_next = Button(ax_next, 'Next')
            b_next.on_clicked(callback.next)
            b_prev = Button(ax_prev, 'Previous')
            b_prev.on_clicked(callback.prev)

        show_or_save_figure(fig, save_path, dpi)
