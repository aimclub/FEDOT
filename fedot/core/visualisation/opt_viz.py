from __future__ import annotations

import itertools
import os
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from glob import glob
from os import remove
from time import time
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from fedot.core.visualisation.opt_history.fitness_box import visualise_fitness_box
from fedot.core.visualisation.opt_history.fitness_line import visualize_fitness_line, visualize_fitness_line_interactive
from fedot.core.visualisation.opt_history.operations_animated_bar import visualize_operations_animated_bar
from fedot.core.visualisation.opt_history.operations_kde import visualize_operations_kde
from fedot.utilities.requirements_notificator import warn_requirement

try:
    import PIL
    from PIL import Image
except ModuleNotFoundError:
    warn_requirement('Pillow')
    PIL = None

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.pipelines.convert import pipeline_template_as_nx_graph
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.graph_viz import GraphVisualiser

if TYPE_CHECKING:
    pass


class PlotTypesEnum(Enum):
    fitness_line = partial(visualize_fitness_line)
    fitness_line_interactive = partial(visualize_fitness_line_interactive)
    fitness_box = partial(visualise_fitness_box)
    operations_kde = partial(visualize_operations_kde)
    operations_animated_bar = partial(visualize_operations_animated_bar)

    @classmethod
    def member_names(cls):
        return cls._member_names_


class PipelineEvolutionVisualiser:
    def __init__(self, history):
        default_data_dir = default_fedot_data_dir()
        self.history = history
        for function in PlotTypesEnum:
            self.__setattr__(function.name, partial(function.value, history=self.history))

        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        if 'composing_history' not in os.listdir(default_data_dir):
            os.mkdir(self.temp_path)
        self.log = default_log(self)
        self.pipelines_imgs = []
        self.convergence_imgs = []
        self.best_pipelines_imgs = []
        self.merged_imgs = []
        self.graph_visualizer = GraphVisualiser()

    def __call__(self, plot_type: Union[PlotTypesEnum, str] = PlotTypesEnum.fitness_box,
                 save_path: Optional[Union[os.PathLike, str]] = None, dpi: int = 300,
                 best_fraction: Optional[float] = None, show_fitness: bool = True, per_time: bool = True,
                 use_tags: bool = True):
        """ Visualizes fitness values or operations used across generations.

        :param plot_type: visualization to show. Expected values are listed in
            'fedot.core.visualisation.opt_viz.PlotTypesEnum'.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed. Essential for animations.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of individuals with the best fitness per generation. The value should be in the
            interval (0, 1]. The other individuals are filtered out. The fraction will also be mentioned on the plot.
        :param show_fitness: if False, visualizations that support this parameter will not display fitness.
        :param per_time: Shows time axis instead of generations axis. Currently, supported for plot types:
            'show_fitness_line', 'show_fitness_line_interactive'.
        :param use_tags: if True (default), all operations in the history are colored and grouped based on FEDOT
            repo tags. If False, operations are not grouped, colors are picked by fixed colormap for every history
            independently.
        """

        def check_args_constraints():
            nonlocal per_time
            # Check supported cases for `best_fraction`.
            if best_fraction is not None and \
                    (best_fraction <= 0 or best_fraction > 1):
                raise ValueError('`best_fraction` parameter should be in the interval (0, 1].')
            # Check supported cases for show_fitness == False.
            if not show_fitness and plot_type is not PlotTypesEnum.operations_animated_bar:
                default_log().warning(
                    f'Argument `show_fitness` is not supported for "{plot_type.name}". It is ignored.')
            # Check plot_type-specific cases
            if plot_type in (PlotTypesEnum.fitness_line, PlotTypesEnum.fitness_line_interactive) and \
                    per_time and self.history.individuals[0][0].metadata.get('evaluation_time_iso') is None:
                default_log().warning('Evaluation time not found in optimization history. '
                                      'Showing fitness plot per generations...')
                per_time = False
            elif plot_type is PlotTypesEnum.operations_animated_bar:
                if not save_path:
                    raise ValueError('Argument `save_path` is required to save the animation.')

        if isinstance(plot_type, str):
            try:
                plot_type = PlotTypesEnum[plot_type]
            except KeyError:
                raise NotImplementedError(
                    f'Visualization "{plot_type}" is not supported. Expected values: '
                    f'{", ".join(PlotTypesEnum.member_names())}.')

        check_args_constraints()

        default_log().info(
            'Visualizing optimization history... It may take some time, depending on the history size.')

        if plot_type is PlotTypesEnum.fitness_line:
            visualize_fitness_line(self.history, per_time, save_path, dpi)
        elif plot_type is PlotTypesEnum.fitness_line_interactive:
            visualize_fitness_line_interactive(self.history, per_time, save_path, dpi, use_tags)
        elif plot_type is PlotTypesEnum.fitness_box:
            visualise_fitness_box(self.history, save_path, dpi, best_fraction)
        elif plot_type is PlotTypesEnum.operations_kde:
            visualize_operations_kde(self.history, save_path, dpi, best_fraction, use_tags)
        elif plot_type is PlotTypesEnum.operations_animated_bar:
            visualize_operations_animated_bar(self.history, save_path, dpi, best_fraction, show_fitness, use_tags)
        else:
            raise NotImplementedError(f'Oops, plot type {plot_type.name} has no function to show!')

    def _visualise_pipelines(self, pipelines, fitnesses):
        fitnesses = deepcopy(fitnesses)
        last_best_pipeline = pipelines[0]
        prev_fit = fitnesses[0]
        fig = plt.figure(figsize=(10, 10))
        for ch_id, pipeline in enumerate(pipelines):
            self.graph_visualizer.draw_nx_dag(pipeline,
                                              in_graph_converter_function=pipeline_template_as_nx_graph)
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.pipelines_imgs.append(img)
            plt.clf()
            if fitnesses[ch_id] > prev_fit:
                fitnesses[ch_id] = prev_fit
            else:
                last_best_pipeline = pipeline
            prev_fit = fitnesses[ch_id]
            plt.clf()
            self.graph_visualizer.draw_nx_dag(last_best_pipeline,
                                              in_graph_converter_function=pipeline_template_as_nx_graph)
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.best_pipelines_imgs.append(img)
            plt.clf()
        plt.close('all')

    def _visualise_convergence(self, fitness_history):
        fitness_history = deepcopy(fitness_history)
        prev_fit = fitness_history[0]
        for fit_id, fit in enumerate(fitness_history):
            if fit > prev_fit:
                fitness_history[fit_id] = prev_fit
            prev_fit = fitness_history[fit_id]
        ts_set = list(range(len(fitness_history)))
        df = pd.DataFrame(
            {'ts': ts_set, 'fitness': [-f.value for f in fitness_history]})

        fig = plt.figure(figsize=(10, 10))
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        for ts in ts_set:
            plt.plot(df['ts'], df['fitness'], label='Composer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best metric', fontsize=18)
            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.convergence_imgs.append(img)
            plt.clf()
        plt.close('all')

    def visualise_history(self, history):
        try:
            self._clean(with_gif=True)
            all_historical_fitness = history.all_historical_quality
            self._visualise_pipelines(history.historical_pipelines, all_historical_fitness)
            self._visualise_convergence(all_historical_fitness)
            self._merge_images()
            self._combine_gifs()
            self._clean()
        except Exception as ex:
            self.log.error(f'Visualisation failed with {ex}')

    def _merge_images(self):
        from PIL import Image

        for i in range(1, len(self.pipelines_imgs)):
            im1 = self.pipelines_imgs[i]
            im2 = self.best_pipelines_imgs[i]
            im3 = self.convergence_imgs[i]
            imgs = (im1, im2, im3)
            merged = np.concatenate(imgs, axis=1)
            self.merged_imgs.append(Image.fromarray(np.uint8(merged)))

    def _combine_gifs(self):
        path = f'{self.temp_path}final_{str(time())}.gif'
        imgs = self.merged_imgs[1:]
        self.merged_imgs[0].save(path, save_all=True, append_images=imgs,
                                 optimize=False, duration=0.5, loop=0)

    def _clean(self, with_gif=False):
        files = glob(f'{self.temp_path}*.png')
        if with_gif:
            files += glob(f'{self.temp_path}*.gif')
        for file in files:
            remove(file)

    def create_gif_using_images(self, gif_path: str, files: List[str]):
        from imageio import get_writer, imread

        with get_writer(gif_path, mode='I', duration=0.5) as writer:
            for filename in files:
                image = imread(filename)
                writer.append_data(image)

    def objectives_lists(self, individuals: List[Any], objectives_numbers: Tuple[int] = None):
        num_of_objectives = len(objectives_numbers) if objectives_numbers else len(individuals[0].fitness.values)
        objectives_numbers = objectives_numbers if objectives_numbers else [i for i in range(num_of_objectives)]
        objectives_values_set = [[] for _ in range(num_of_objectives)]
        for obj_num in range(num_of_objectives):
            for individual in individuals:
                value = individual.fitness.values[objectives_numbers[obj_num]]
                objectives_values_set[obj_num].append(value if value > 0 else -value)
        return objectives_values_set

    def extract_objectives(self, individuals: List[List[Any]], objectives_numbers: Tuple[int] = None,
                           transform_from_minimization=True):
        if not objectives_numbers:
            objectives_numbers = [i for i in range(len(individuals[0][0].fitness.values))]
        all_inds = list(itertools.chain(*individuals))
        all_objectives = [[ind.fitness.values[i] for ind in all_inds] for i in objectives_numbers]
        if transform_from_minimization:
            transformed_objectives = []
            for obj_values in all_objectives:
                are_objectives_positive = all(np.array(obj_values) > 0)
                if not are_objectives_positive:
                    transformed_obj_values = list(np.array(obj_values) * (-1))
                else:
                    transformed_obj_values = obj_values
                transformed_objectives.append(transformed_obj_values)
        else:
            transformed_objectives = all_objectives
        return transformed_objectives

    def create_boxplot(self, individuals: List[Any], generation_num: int = None,
                       objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'), file_name: str = 'obj_boxplots.png',
                       folder: str = None, y_limits: Tuple[float] = None):
        folder = f'{self.temp_path}/boxplots' if folder is None else folder
        fig, ax = plt.subplots()
        ax.set_title(f'Generation: {generation_num}', fontsize=15)
        objectives = self.objectives_lists(individuals)
        df_objectives = pd.DataFrame({objectives_names[i]: objectives[i] for i in range(len(objectives))})
        sns.boxplot(data=df_objectives, palette="Blues")
        if y_limits:
            plt.ylim(y_limits[0], y_limits[1])
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')
        if not os.path.isdir(f'{folder}'):
            os.mkdir(f'{folder}')
        path = f'{folder}/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    def boxplots_gif_create(self, individuals: List[List[Any]],
                            objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'),
                            folder: str = None):
        objectives = self.extract_objectives(individuals)
        objectives = list(itertools.chain(*objectives))
        min_y, max_y = min(objectives), max(objectives)
        files = []
        folder = f'{self.temp_path}' if folder is None else folder
        for generation_num, individuals_in_genaration in enumerate(individuals):
            file_name = f'{generation_num}.png'
            self.create_boxplot(individuals_in_genaration, generation_num, objectives_names,
                                file_name=file_name, folder=folder, y_limits=(min_y, max_y))
            files.append(f'{folder}/{file_name}')
        self.create_gif_using_images(gif_path=f'{folder}/boxplots_history.gif', files=files)
        for file in files:
            remove(file)
        plt.cla()
        plt.clf()
        plt.close('all')

    def visualise_pareto(self, front: Sequence[Individual],
                         objectives_numbers: Tuple[int, int] = (0, 1),
                         objectives_names: Sequence[str] = ('ROC-AUC', 'Complexity'),
                         file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                         folder: str = f'../../tmp/pareto',
                         generation_num: int = None,
                         individuals: Sequence[Individual] = None,
                         minmax_x: List[float] = None,
                         minmax_y: List[float] = None):

        pareto_obj_first, pareto_obj_second = [], []
        for ind in front:
            fit_first = ind.fitness.values[objectives_numbers[0]]
            pareto_obj_first.append(abs(fit_first))
            fit_second = ind.fitness.values[objectives_numbers[1]]
            pareto_obj_second.append(abs(fit_second))

        fig, ax = plt.subplots()

        if individuals is not None:
            obj_first, obj_second = [], []
            for ind in individuals:
                fit_first = ind.fitness.values[objectives_numbers[0]]
                obj_first.append(abs(fit_first))
                fit_second = ind.fitness.values[objectives_numbers[1]]
                obj_second.append(abs(fit_second))
            ax.scatter(obj_first, obj_second, c='green')

        ax.scatter(pareto_obj_first, pareto_obj_second, c='red')
        plt.plot(pareto_obj_first, pareto_obj_second, color='r')

        if generation_num is not None:
            ax.set_title(f'Pareto frontier, Generation: {generation_num}', fontsize=15)
        else:
            ax.set_title('Pareto frontier', fontsize=15)
        plt.xlabel(objectives_names[0], fontsize=15)
        plt.ylabel(objectives_names[1], fontsize=15)

        if minmax_x is not None:
            plt.xlim(minmax_x[0], minmax_x[1])
        if minmax_y is not None:
            plt.ylim(minmax_y[0], minmax_y[1])
        fig.set_figwidth(8)
        fig.set_figheight(8)
        if save:
            if not os.path.isdir('../../tmp'):
                os.mkdir('../../tmp')
            if not os.path.isdir(f'{folder}'):
                os.mkdir(f'{folder}')

            path = f'{folder}/{file_name}'
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close('all')

    def pareto_gif_create(self, pareto_fronts: List[List[Any]], individuals: List[List[Any]] = None,
                          objectives_numbers: Tuple[int] = (1, 0),
                          objectives_names: Tuple[str] = ('Complexity', 'ROC-AUC')):
        files = []
        array_for_analysis = individuals if individuals else pareto_fronts
        all_objectives = self.extract_objectives(array_for_analysis, objectives_numbers)
        min_x, max_x = min(all_objectives[0]) - 0.01, max(all_objectives[0]) + 0.01
        min_y, max_y = min(all_objectives[1]) - 0.01, max(all_objectives[1]) + 0.01
        folder = f'{self.temp_path}'
        for i, front in enumerate(pareto_fronts):
            file_name = f'pareto{i}.png'
            self.visualise_pareto(front, file_name=file_name, save=True, show=False,
                                  folder=folder, generation_num=i, individuals=individuals[i],
                                  minmax_x=[min_x, max_x], minmax_y=[min_y, max_y],
                                  objectives_numbers=objectives_numbers,
                                  objectives_names=objectives_names)
            files.append(f'{folder}/{file_name}')

        self.create_gif_using_images(gif_path=f'{folder}/pareto_history.gif', files=files)
        for file in files:
            remove(file)


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
