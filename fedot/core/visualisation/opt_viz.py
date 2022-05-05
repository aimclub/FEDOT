import itertools
import math
import os
from copy import deepcopy
from glob import glob
from os import remove
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.common import flatten
import seaborn as sns
from PIL import Image
from deap import tools
from imageio import get_writer, imread
import matplotlib.pyplot as plt

from fedot.core.log import Log, default_log
from fedot.core.pipelines.convert import pipeline_template_as_nx_graph
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_opt_node_tag
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.graph_viz import GraphVisualiser


class PipelineEvolutionVisualiser:

    def __init__(self, log: Optional[Log] = None):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        if 'composing_history' not in os.listdir(default_data_dir):
            os.mkdir(self.temp_path)
        self.log = log or default_log(__name__)
        self.pipelines_imgs = []
        self.convergence_imgs = []
        self.best_pipelines_imgs = []
        self.merged_imgs = []
        self.graph_visualizer = GraphVisualiser(log=log)

    def _visualise_pipelines(self, pipelines, fitnesses):
        fitnesses = deepcopy(fitnesses)
        last_best_pipeline = pipelines[0]
        prev_fit = fitnesses[0]
        fig = plt.figure(figsize=(10, 10))
        for ch_id, pipeline in enumerate(pipelines):
            self.graph_visualizer.draw_single_graph(pipeline,
                                                    title='Current graph',
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
            self.graph_visualizer.draw_single_graph(last_best_pipeline,
                                                    title=f'Best graph after {round(ch_id)} evals',
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
            {'ts': ts_set, 'fitness': [-f for f in fitness_history]})

        fig = plt.figure(figsize=(10, 10))
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        for ts in ts_set:
            plt.plot(df['ts'], df['fitness'])
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best metric', fontsize=18)
            plt.axvline(x=ts, color='black')
            plt.legend(loc='upper left')
            fig.canvas.draw()
            img = figure_to_array(fig)
            self.convergence_imgs.append(img)
            plt.clf()
        plt.close('all')

    @staticmethod
    def visualise_fitness_by_generations(history, save_path_to_file: str):
        """ Visualizes fitness values across generations
        :param history: OptHistory
        :param save_path_to_file: path where to save visualization. If set, then the image will be saved,
        and if not, it will be displayed """
        # Get list of generations numbers per fitness result
        generations = []
        for gen_num in range(len(history.historical_fitness)):
            num_of_ind_in_gen = len(history.historical_fitness[gen_num])
            generations.append([gen_num] * num_of_ind_in_gen)
        generations = list(flatten(generations))

        # Visualize
        fitness = [abs(fitness.value) for fitness in history.all_historical_fitness]
        fig, ax = plt.subplots(figsize=(15, 10))

        # Get color palette for fitness. The lower the fitness value, the brighter the green color
        palette = sns.light_palette("seagreen", n_colors=len(history.historical_fitness))
        min_fitnesses = [min(i) for i in history.historical_fitness]
        min_fitnesses.sort(reverse=True)
        colors = [palette[min_fitnesses.index(min(i))] for i in history.historical_fitness]

        sns.boxplot(x=generations, y=fitness, palette=colors)
        ax.set_title('Fitness by generations', fontdict={'fontsize': 22})
        ax.set_xticklabels(range(len(history.historical_fitness)))
        ax.set_xlabel(xlabel=f'generations', fontsize=20)
        ax.set_ylabel(ylabel=f'fitness score', fontsize=20)

        if not save_path_to_file:
            plt.show()
        else:
            plt.savefig(save_path_to_file)
            plt.close()

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

    def visualise_pareto(self, archive: Any, objectives_numbers: Tuple[int, int] = (0, 1),
                         objectives_names: Tuple[str] = ('ROC-AUC', 'Complexity'),
                         file_name: str = 'result_pareto.png', show: bool = False, save: bool = True,
                         folder: str = f'../../tmp/pareto',
                         generation_num: int = None, individuals: List[Any] = None, minmax_x: List[float] = None,
                         minmax_y: List[float] = None):

        pareto_obj_first, pareto_obj_second = [], []
        for i in range(len(archive)):
            fit_first = archive[i].fitness.values[objectives_numbers[0]]
            pareto_obj_first.append(abs(fit_first))
            fit_second = archive[i].fitness.values[objectives_numbers[1]]
            pareto_obj_second.append(abs(fit_second))

        fig, ax = plt.subplots()

        if individuals is not None:
            obj_first, obj_second = [], []
            for i in range(len(individuals)):
                fit_first = individuals[i].fitness.values[objectives_numbers[0]]
                obj_first.append(abs(fit_first))
                fit_second = individuals[i].fitness.values[objectives_numbers[1]]
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

    def pareto_gif_create(self, pareto_fronts: List[tools.ParetoFront], individuals: List[List[Any]] = None,
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

    @staticmethod
    def __get_history_dataframe(history: 'OptHistory', tags_model: Optional[List[str]] = None,
                                tags_data: Optional[List[str]] = None, tag_column_name: str = 'Operation',
                                generation_column_name: str = 'Generation', individual_column_name: str = 'Individual',
                                fitness_column_name: str = 'Fitness', n_best: Optional[float] = None):
        history_data = {
            generation_column_name: [],
            tag_column_name: [],
            individual_column_name: [],
            fitness_column_name: []
        }

        for gen_num, gen in enumerate(history.individuals):
            for ind in gen:
                for node in ind.graph.nodes:
                    history_data[generation_column_name].append(gen_num)
                    history_data[tag_column_name].append(
                        get_opt_node_tag(node, tags_model=tags_model, tags_data=tags_data)
                    )
                    history_data[individual_column_name].append(ind.uid)
                    history_data[fitness_column_name].append(abs(ind.fitness))

        df_history = pd.DataFrame.from_dict(history_data)

        if n_best is not None:
            generation_sizes = df_history.groupby(generation_column_name)[individual_column_name].nunique()

            df_individuals = df_history[[generation_column_name, individual_column_name, fitness_column_name]] \
                .drop_duplicates(ignore_index=True)

            df_individuals['rank_per_generation'] = df_individuals.sort_values(fitness_column_name, ascending=False). \
                groupby(generation_column_name).cumcount()

            best_individuals = df_individuals[df_individuals.apply(
                lambda row: row['rank_per_generation'] < generation_sizes[row[generation_column_name]] * n_best,
                axis=1)][individual_column_name]

            df_history = df_history[df_history[individual_column_name].isin(best_individuals)]

        return df_history

    def visualize_operations_kde(self, history: 'OptHistory', save_path_to_file: Optional[str] = None,
                                 tags_model: Optional[List[str]] = None, tags_data: Optional[List[str]] = None,
                                 n_best: Optional[float] = None):
        # TODO: Docstring
        tags_model = tags_model or OperationTypesRepository.DEFAULT_MODEL_TAGS
        tags_data = tags_data or OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS

        tags_all = [*tags_model, *tags_data]

        tag_column_name = 'Operation'
        generation_column_name = 'Generation'
        individual_column_name = 'Individual'
        fitness_column_name = 'Fitness'

        df_history = self.__get_history_dataframe(
            history, tags_model, tags_data, tag_column_name, generation_column_name, individual_column_name,
            fitness_column_name, n_best
        )
        tags_found = df_history[tag_column_name].unique()

        plot = sns.displot(
            data=df_history,
            x=generation_column_name,
            hue=tag_column_name,
            hue_order=[t for t in tags_all if t in tags_found],
            kind='kde',
            clip=(0, max(df_history[generation_column_name])),
            multiple='fill',
            palette='Set2',
        )
        fig = plot.figure
        fig.set_dpi(110)
        fig.set_facecolor('w')

        if save_path_to_file:
            fig.savefig(save_path_to_file, dpi=300)
            plt.close()


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
