from __future__ import annotations

import itertools
import os
from copy import deepcopy
from datetime import datetime
from enum import Enum, auto
from glob import glob
from os import remove
from pathlib import Path
from textwrap import wrap
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, cm, pyplot as plt, ticker
from matplotlib.colors import Normalize
from matplotlib.widgets import Button

from fedot.utilities.requirements_notificator import warn_requirement

try:
    import PIL
    from PIL import Image
except ModuleNotFoundError:
    warn_requirement('Pillow')
    PIL = None

from fedot.core.log import default_log
from fedot.core.optimisers.fitness import null_fitness
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.pipelines.convert import pipeline_template_as_nx_graph
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_opt_node_tag
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.graph_viz import GraphVisualiser

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history import OptHistory


def with_alternate_matplotlib_backend(func):
    def wrapper(*args, **kwargs):
        default_mpl_backend = mpl.get_backend()
        try:
            mpl.use('TKAgg')
            return func(*args, **kwargs)
        except ImportError as e:
            default_log('Requirements').warning(e)
        finally:
            mpl.use(default_mpl_backend)

    return wrapper


class PlotTypesEnum(Enum):
    fitness_line = auto()
    fitness_line_interactive = auto()
    fitness_box = auto()
    operations_kde = auto()
    operations_animated_bar = auto()

    @classmethod
    def member_names(cls):
        return cls._member_names_


class PipelineEvolutionVisualiser:

    def __init__(self):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        if 'composing_history' not in os.listdir(default_data_dir):
            os.mkdir(self.temp_path)
        self.log = default_log(self)
        self.pipelines_imgs = []
        self.convergence_imgs = []
        self.best_pipelines_imgs = []
        self.merged_imgs = []
        self.graph_visualizer = GraphVisualiser()

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

    def __show_or_save_figure(self, figure: plt.Figure, save_path: Optional[Union[os.PathLike, str]], dpi: int = 300):
        if not save_path:
            plt.show()
        else:
            save_path = Path(save_path)
            if not save_path.is_absolute():
                save_path = Path.cwd().joinpath(save_path)
            figure.savefig(save_path, dpi=dpi)
            self.log.info(f'The figure was saved to "{save_path}".')
            plt.close()

    @staticmethod
    def __plot_fitness_line_per_generations(axis: plt.Axes, generations, label: Optional[str] = None) \
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
        return best_individuals

    @staticmethod
    def __plot_fitness_line_per_time(axis: plt.Axes, generations: List[List[Individual]], label: Optional[str] = None,
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

    @staticmethod
    def __setup_fitness_plot(axis: plt.Axes, xlabel: str, title: Optional[str] = None, with_legend: bool = False):
        if axis is None:
            fig, axis = plt.subplots()

        if with_legend:
            axis.legend()
        axis.set_ylabel('Fitness')
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis='y')

    def visualize_fitness_line(self, history: OptHistory, per_time: bool = True,
                               save_path: Optional[Union[os.PathLike, str]] = None, dpi: int = 300):
        ax = plt.gca()
        if per_time:
            xlabel = 'Time, s'
            self.__plot_fitness_line_per_time(ax, history.individuals)
        else:
            xlabel = 'Generation'
            self.__plot_fitness_line_per_generations(ax, history.individuals)
        self.__setup_fitness_plot(ax, xlabel)
        self.__show_or_save_figure(plt.gcf(), save_path, dpi)

    @with_alternate_matplotlib_backend
    def visualize_fitness_line_interactive(self, history: OptHistory, per_time: bool = True,
                                           save_path: Optional[Union[os.PathLike, str]] = None, dpi: int = 300,
                                           use_tags: bool = True):
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        ax_fitness, ax_graph = axes

        if per_time:
            x_label = 'Time, s'
            x_template = 'time {} s'
            plot_func = self.__plot_fitness_line_per_time
        else:
            x_label = 'Generation'
            x_template = 'generation {}'
            plot_func = self.__plot_fitness_line_per_generations

        best_individuals = plot_func(ax_fitness, history.individuals)
        self.__setup_fitness_plot(ax_fitness, x_label)

        ax_graph.axis('off')

        class InteractivePlot:
            temp_path = Path(default_fedot_data_dir(), 'current_graph.png')

            def __init__(self, best_individuals: Dict[int, Individual]):
                self.best_x: List[int] = list(best_individuals.keys())
                self.best_individuals: List[Individual] = list(best_individuals.values())
                self.index: int = len(best_individuals) - 1
                self.time_line = ax_fitness.axvline(self.best_x[self.index], color='r', alpha=0.7)
                self.graph_images: List[np.ndarray] = []
                self.generate_graph_images()
                self.update_graph()

            def generate_graph_images(self):
                for ind in self.best_individuals:
                    graph = ind.graph
                    if use_tags:
                        graph = PipelineAdapter().restore(ind.graph)
                    graph.show(self.temp_path)
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

        self.__show_or_save_figure(fig, save_path, dpi)

    @staticmethod
    def __get_history_dataframe(history: OptHistory, tags_model: Optional[List[str]] = None,
                                tags_data: Optional[List[str]] = None, best_fraction: Optional[float] = None,
                                get_tags: bool = True):
        history_data = {
            'generation': [],
            'individual': [],
            'fitness': [],
            'node': [],
        }
        if get_tags:
            history_data['tag'] = []

        uid_counts = {}  # Resolving individuals with the same uid
        for gen_num, gen in enumerate(history.individuals):
            for ind in gen:
                uid_counts[ind.uid] = uid_counts.get(ind.uid, -1) + 1
                for node in ind.graph.nodes:
                    history_data['generation'].append(gen_num)
                    history_data['individual'].append('_'.join([ind.uid, str(uid_counts[ind.uid])]))
                    fitness = abs(ind.fitness.value)
                    history_data['fitness'].append(fitness)
                    history_data['node'].append(str(node))
                    if not get_tags:
                        continue
                    history_data['tag'].append(get_opt_node_tag(str(node), tags_model=tags_model, tags_data=tags_data))

        df_history = pd.DataFrame.from_dict(history_data)

        if best_fraction is not None:
            generation_sizes = df_history.groupby('generation')['individual'].nunique()

            df_individuals = df_history[['generation', 'individual', 'fitness']] \
                .drop_duplicates(ignore_index=True)

            df_individuals['rank_per_generation'] = df_individuals.sort_values('fitness', ascending=False). \
                groupby('generation').cumcount()

            best_individuals = df_individuals[
                df_individuals.apply(
                    lambda row: row['rank_per_generation'] < generation_sizes[row['generation']] * best_fraction,
                    axis='columns'
                )
            ]['individual']

            df_history = df_history[df_history['individual'].isin(best_individuals)]

        return df_history

    def visualise_fitness_box(self, history: OptHistory, save_path: Optional[Union[os.PathLike, str]] = None,
                              dpi: int = 300, best_fraction: Optional[float] = None):
        """ Visualizes fitness values across generations in the form of boxplot.

        :param history: OptHistory.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param dpi: DPI if the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the visualization.
            Must be in the interval (0, 1].
        """
        df_history = self.__get_history_dataframe(history, get_tags=False, best_fraction=best_fraction)
        columns_needed = ['generation', 'individual', 'fitness']
        df_history = df_history[columns_needed].drop_duplicates(ignore_index=True)
        # Get color palette by mean fitness per generation
        fitness = df_history.groupby('generation')['fitness'].mean()
        fitness = (fitness - min(fitness)) / (max(fitness) - min(fitness))
        colormap = sns.color_palette('YlOrRd', as_cmap=True)

        plot = sns.boxplot(data=df_history, x='generation', y='fitness', palette=fitness.map(colormap))
        fig = plot.figure
        fig.set_dpi(dpi)
        fig.set_facecolor('w')
        ax = plt.gca()

        ax.set_title('Fitness by generations')
        ax.set_xlabel('Generation')
        # Set ticks for every 5 generation if there's more than 10 generations.
        if len(history.individuals) > 10:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.grid(True)
        str_fraction_of_pipelines = 'all' if best_fraction is None else f'top {best_fraction * 100}% of'
        ax.set_ylabel(f'Fitness of {str_fraction_of_pipelines} generation pipelines')
        ax.yaxis.grid(True)

        self.__show_or_save_figure(fig, save_path, dpi)

    def visualize_operations_kde(self, history: OptHistory, save_path: Optional[Union[os.PathLike, str]] = None,
                                 dpi: int = 300, best_fraction: Optional[float] = None, use_tags: bool = True,
                                 tags_model: Optional[List[str]] = None, tags_data: Optional[List[str]] = None):
        """ Visualizes operations used across generations in the form of KDE.

        :param history: OptHistory.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the visualization.
            Must be in the interval (0, 1].
        :param use_tags: if True (default), all operations in the history are colored and grouped based on FEDOT
            repo tags. If False, operations are not grouped, colors are picked by fixed colormap for every history
            independently.
        :param tags_model: tags for OperationTypesRepository('model') to map the history operations.
            The later the tag, the higher its priority in case of intersection.
        :param tags_data: tags for OperationTypesRepository('data_operation') to map the history operations.
            The later the tag, the higher its priority in case of intersection.
        """

        tags_model = tags_model or OperationTypesRepository.DEFAULT_MODEL_TAGS
        tags_data = tags_data or OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS

        tags_all = [*tags_model, *tags_data]

        generation_column_name = 'Generation'
        operation_column_name = 'Operation'
        column_for_operation = 'tag' if use_tags else 'node'

        df_history = self.__get_history_dataframe(history, tags_model, tags_data, best_fraction, use_tags)
        df_history = df_history.rename({'generation': generation_column_name,
                                        column_for_operation: operation_column_name}, axis='columns')
        operations_found = df_history[operation_column_name].unique()
        if use_tags:
            operations_found = [t for t in tags_all if t in operations_found]
            nodes_per_tag = df_history.groupby(operation_column_name)['node'].unique()
            legend = [get_description_of_operations_by_tag(tag, nodes_per_tag[tag]) for tag in operations_found]
            palette = get_palette_based_on_default_tags()
        else:
            legend = operations_found
            palette = sns.color_palette('tab10', n_colors=len(operations_found))

        plot = sns.displot(
            data=df_history,
            x=generation_column_name,
            hue=operation_column_name,
            hue_order=operations_found,
            kind='kde',
            clip=(0, max(df_history[generation_column_name])),
            multiple='fill',
            palette=palette
        )

        for text, new_text in zip(plot.legend.texts, legend):
            text.set_text(new_text)

        fig = plot.figure
        fig.set_dpi(dpi)
        fig.set_facecolor('w')
        ax = plt.gca()
        str_fraction_of_pipelines = 'all' if best_fraction is None else f'top {best_fraction * 100}% of'
        ax.set_ylabel(f'Fraction in {str_fraction_of_pipelines} generation pipelines')

        self.__show_or_save_figure(fig, save_path, dpi)

    def visualize_operations_animated_bar(self, history: OptHistory, save_path: Union[os.PathLike, str],
                                          dpi: int = 300, best_fraction: Optional[float] = None,
                                          show_fitness_color: bool = True, use_tags: bool = True,
                                          tags_model: Optional[List[str]] = None,
                                          tags_data: Optional[List[str]] = None):
        """ Visualizes operations used across generations in the form of animated bar plot.

        :param history: OptHistory instance.
        :param save_path: path to save the visualization.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the visualization.
            Must be in the interval (0, 1].
        :param show_fitness_color: if False, the bar colors will not correspond to fitness.
        :param use_tags: if True (default), all operations in the history are colored and grouped based on FEDOT
            repo tags. If False, operations are not grouped, colors are picked by fixed colormap for every history
            independently.
        :param tags_model: tags for OperationTypesRepository('model') to map the history operations.
            The later the tag, the higher its priority in case of intersection.
        :param tags_data: tags for OperationTypesRepository('data_operation') to map the history operations.
            The later the tag, the higher its priority in case of intersection.
        """

        def interpolate_points(point_1, point_2, smoothness=18, power=4) -> List[np.array]:
            t_interp = np.linspace(0, 1, smoothness)
            point_1, point_2 = np.array(point_1), np.array(point_2)
            return [point_1 * (1 - t ** power) + point_2 * t ** power for t in t_interp]

        def smoothen_frames_data(data: Sequence[Sequence['ArrayLike']], smoothness=18, power=4) -> List[np.array]:
            final_frames = []
            for initial_frame in range(len(data) - 1):
                final_frames += interpolate_points(data[initial_frame], data[initial_frame + 1], smoothness, power)
            # final frame interpolates into itself
            final_frames += interpolate_points(data[-1], data[-1], smoothness, power)

            return final_frames

        def animate(frame_num):
            frame_count = bar_data[frame_num]
            frame_color = bar_color[frame_num] if show_fitness_color else None
            frame_title = bar_title[frame_num]

            plt.title(frame_title)
            for bar_num in range(len(bars)):
                bars[bar_num].set_width(frame_count[bar_num])
                if not show_fitness_color:
                    continue
                bars[bar_num].set_facecolor(frame_color[bar_num])

        save_path = Path(save_path)
        if save_path.suffix not in ['.gif', '.mp4']:
            raise ValueError('A correct file extension (".mp4" or ".gif") should be set to save the animation.')

        animation_frames_per_step = 18
        animation_interval_between_frames_ms = 40
        animation_interpolation_power = 4
        fitness_colormap = cm.get_cmap('YlOrRd')

        tags_model = tags_model or OperationTypesRepository.DEFAULT_MODEL_TAGS
        tags_data = tags_data or OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS

        tags_all = [*tags_model, *tags_data]

        generation_column_name = 'Generation'
        fitness_column_name = 'Fitness'
        operation_column_name = 'Operation'
        column_for_operation = 'tag' if use_tags else 'node'

        df_history = self.__get_history_dataframe(history, tags_model, tags_data, best_fraction, use_tags)
        df_history = df_history.rename({
            'generation': generation_column_name,
            'fitness': fitness_column_name,
            column_for_operation: operation_column_name,
        }, axis='columns')
        operations_found = df_history[operation_column_name].unique()
        if use_tags:
            operations_found = [tag for tag in tags_all if tag in operations_found]
            nodes_per_tag = df_history.groupby(operation_column_name)['node'].unique()
            bars_labels = [get_description_of_operations_by_tag(t, nodes_per_tag[t], 22) for t in operations_found]
            no_fitness_palette = get_palette_based_on_default_tags()
        else:
            bars_labels = operations_found
            no_fitness_palette = sns.color_palette('tab10', n_colors=len(operations_found))
            no_fitness_palette = {o: no_fitness_palette[i] for i, o in enumerate(operations_found)}

        # Getting normed fraction of individuals  per generation that contain operations given.
        generation_sizes = df_history.groupby(generation_column_name)['individual'].nunique()
        operations_with_individuals_count = df_history.groupby(
            [generation_column_name, operation_column_name],
            as_index=False
        ).aggregate({'individual': 'nunique'})
        operations_with_individuals_count['individual'] = operations_with_individuals_count.apply(
            lambda row: row['individual'] / generation_sizes[row[generation_column_name]],
            axis='columns')

        if show_fitness_color:
            # Getting fitness per individual with the list of contained operations in the form of
            # '.operation_1.operation_2. ... .operation_n.'
            individuals_fitness = df_history[[generation_column_name, 'individual', fitness_column_name]] \
                .drop_duplicates()
            individuals_fitness['operations'] = individuals_fitness.apply(
                lambda row: '.{}.'.format('.'.join(
                    df_history[
                        (df_history[generation_column_name] == row[generation_column_name]) &
                        (df_history['individual'] == row['individual'])
                        ][operation_column_name])),
                axis='columns')
            # Getting mean fitness of individuals with the operations given.
            operations_with_individuals_count[fitness_column_name] = operations_with_individuals_count.apply(
                lambda row: individuals_fitness[
                    (individuals_fitness[generation_column_name] == row[generation_column_name]) &
                    (individuals_fitness['operations'].str.contains(f'.{row[operation_column_name]}.'))
                    ][fitness_column_name].mean(),
                axis='columns')
            del individuals_fitness
        # Replacing the initial DataFrame with the processed one
        df_history = operations_with_individuals_count.set_index([generation_column_name, operation_column_name])
        del operations_with_individuals_count

        min_fitness = df_history[fitness_column_name].min() if show_fitness_color else None
        max_fitness = df_history[fitness_column_name].max() if show_fitness_color else None

        generations = generation_sizes.index.unique()
        bar_data = []
        bar_color = []
        # Getting data by tags through all generations and filling with zeroes where no such tag
        for gen_num in generations:
            bar_data.append([df_history.loc[gen_num]['individual'].get(tag, 0) for tag in operations_found])
            if not show_fitness_color:
                continue
            fitnesses = [df_history.loc[gen_num][fitness_column_name].get(tag, 0) for tag in operations_found]
            # Transfer fitness to color
            bar_color.append([
                fitness_colormap((fitness - min_fitness) / (max_fitness - min_fitness)) for fitness in fitnesses])

        bar_data = smoothen_frames_data(bar_data, animation_frames_per_step, animation_interpolation_power)
        title_template = 'Generation {}'
        if best_fraction is not None:
            title_template += f', top {best_fraction * 100}%'
        bar_title = [i for gen_num in generations for i in [title_template.format(gen_num)] * animation_frames_per_step]

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='w')
        if show_fitness_color:
            bar_color = smoothen_frames_data(bar_color, animation_frames_per_step, animation_interpolation_power)
            sm = cm.ScalarMappable(norm=Normalize(min_fitness, max_fitness), cmap=fitness_colormap)
            sm.set_array([])
            fig.colorbar(sm, label=fitness_column_name)

        count = bar_data[0]
        color = bar_color[0] if show_fitness_color else [no_fitness_palette[tag] for tag in operations_found]
        title = bar_title[0]

        label_size = 10
        if any(len(label.split('\n')) > 2 for label in bars_labels):
            label_size = 8

        bars = ax.barh(bars_labels, count, color=color)
        ax.tick_params(axis='y', which='major', labelsize=label_size)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f'Fraction of pipelines containing the operation')
        ax.xaxis.grid(True)
        ax.set_ylabel(operation_column_name)
        ax.invert_yaxis()
        plt.tight_layout()

        if not save_path.is_absolute():
            save_path = Path.cwd().joinpath(save_path)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(bar_data),
            interval=animation_interval_between_frames_ms,
            repeat=True
        )
        ani.save(str(save_path), dpi=dpi)
        self.log.info(f'The animation was saved to "{save_path}".')
        plt.close(fig=fig)


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def get_description_of_operations_by_tag(tag: str, operations_by_tag: List[str], max_line_length: int = 22,
                                         format_tag: str = 'it'):
    def make_text_fancy(text: str):
        return text.replace('_', ' ')

    def format_text(text_to_wrap: str, latex_format_tag: str = 'it') -> str:
        formatted_text = '$\\' + latex_format_tag + '{' + text_to_wrap + '}$'
        formatted_text = formatted_text.replace(' ', '\\;')
        return formatted_text

    def format_wrapped_text(wrapped_text: List[str], part_to_format: str, latex_format_tag: str = 'it') -> List[str]:

        long_text = ''.join(wrapped_text)
        first_tag_pos = long_text.find(part_to_format)
        second_tag_pos = first_tag_pos + len(part_to_format)

        line_len = len(wrapped_text[0])

        first_tag_line = first_tag_pos // line_len
        first_tag_char = first_tag_pos % line_len

        second_tag_line = second_tag_pos // line_len
        second_tag_char = second_tag_pos % line_len

        if first_tag_line == second_tag_line:
            wrapped_text[first_tag_line] = (
                    wrapped_text[first_tag_line][:first_tag_char] +
                    format_text(wrapped_text[first_tag_line][first_tag_char:second_tag_char], latex_format_tag) +
                    wrapped_text[first_tag_line][second_tag_char:]
            )
        else:
            for line in range(first_tag_line + 1, second_tag_line):
                wrapped_text[line] = format_text(wrapped_text[line], latex_format_tag)

            wrapped_text[first_tag_line] = (
                wrapped_text[first_tag_line][:first_tag_char] +
                format_text(wrapped_text[first_tag_line][first_tag_char:], latex_format_tag)
            )
            wrapped_text[second_tag_line] = (
                    format_text(wrapped_text[second_tag_line][:second_tag_char], latex_format_tag) +
                wrapped_text[second_tag_line][second_tag_char:]
            )
        return wrapped_text

    tag = make_text_fancy(tag)
    operations_by_tag = ', '.join(operations_by_tag)
    description = f'{tag}: {operations_by_tag}.'
    description = make_text_fancy(description)
    description = wrap(description, max_line_length)
    description = format_wrapped_text(description, tag, format_tag)
    description = '\n'.join(description)
    return description


def get_palette_based_on_default_tags() -> Dict[str, Tuple[float, float, float]]:
    default_tags = [*OperationTypesRepository.DEFAULT_MODEL_TAGS, *OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]
    p_1 = sns.color_palette('tab20')
    colour_period = 2  # diverge similar nearby colors
    p_1 = [p_1[i // (len(p_1) // colour_period) + i * colour_period % len(p_1)] for i in range(len(p_1))]
    p_2 = sns.color_palette('Set3')
    palette = np.vstack([p_1, p_2])
    palette_map = {tag: palette[i] for i, tag in enumerate(default_tags)}
    palette_map.update({None: 'mediumaquamarine'})
    return palette_map
