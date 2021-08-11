import itertools
import os
from copy import deepcopy
from glob import glob
from os import remove
from time import time
from typing import List, Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from deap import tools
from imageio import get_writer, imread
from matplotlib import pyplot as plt

from fedot.core.pipelines.convert import pipeline_template_as_nx_graph
from fedot.core.log import Log, default_log
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.graph_viz import GraphVisualiser


class PipelineEvolutionVisualiser:

    def __init__(self, log: Log = default_log(__name__)):
        default_data_dir = default_fedot_data_dir()
        self.temp_path = os.path.join(default_data_dir, 'composing_history')
        if 'composing_history' not in os.listdir(default_data_dir):
            os.mkdir(self.temp_path)
        self.log = log
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
            plt.plot(df['ts'], df['fitness'], label='Composer')
            plt.xlabel('Evaluation', fontsize=18)
            plt.ylabel('Best ROC AUC', fontsize=18)
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
            pareto_obj_first.append(fit_first if fit_first > 0 else -fit_first)
            fit_second = archive[i].fitness.values[objectives_numbers[1]]
            pareto_obj_second.append(fit_second if fit_second > 0 else -fit_second)

        fig, ax = plt.subplots()

        if individuals is not None:
            obj_first, obj_second = [], []
            for i in range(len(individuals)):
                fit_first = individuals[i].fitness.values[objectives_numbers[0]]
                obj_first.append(fit_first if fit_first > 0 else -fit_first)
                fit_second = individuals[i].fitness.values[objectives_numbers[1]]
                obj_second.append(fit_second if fit_second > 0 else -fit_second)
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


def figure_to_array(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
