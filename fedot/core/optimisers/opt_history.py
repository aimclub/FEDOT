import csv
import io
import itertools
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Union

from fedot.core.log import default_log
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.utils.population_utils import get_metric_position
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum
from fedot.core.serializers import Serializer
from fedot.core.utils import default_fedot_data_dir
from fedot.core.visualisation.opt_viz import OptHistoryVisualizer


class OptHistory:
    """
    Contains optimization history, convert Pipeline to PipelineTemplate, save history to csv.

    :param objective: contains information about metrics used during optimization.
    """

    def __init__(self, objective: Objective = None):
        self._objective = objective or Objective([])
        self.individuals: List[List[Individual]] = []
        self.archive_history: List[List[Individual]] = []
        self._log = default_log(self)

    def is_empty(self) -> bool:
        return not self.individuals

    def add_to_history(self, individuals: List[Individual]):
        self.individuals.append(individuals)

    def add_to_archive_history(self, individuals: List[Individual]):
        self.archive_history.append(individuals)

    def to_csv(self, save_dir: Optional[os.PathLike] = None, file: os.PathLike = 'history.csv'):
        save_dir = save_dir or default_fedot_data_dir()
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(Path(save_dir, file), 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)

            # Write header
            metric_str = 'metric'
            if self._objective.is_multi_objective:
                metric_str += 's'
            header_row = ['index', 'generation', metric_str, 'quantity_of_operations', 'depth', 'metadata']
            writer.writerow(header_row)

            # Write history rows
            idx = 0
            for gen_num, gen_inds in enumerate(self.individuals):
                for ind_num, ind in enumerate(gen_inds):
                    row = [idx, gen_num, ind.fitness.values, ind.graph.length, ind.graph.depth, ind.metadata]
                    writer.writerow(row)
                    idx += 1

    def save_current_results(self, save_dir: os.PathLike):
        # Create folder if it's not exists
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            self._log.info(f"Created directory for saving optimization history: {save_dir}")

        try:
            last_gen_id = len(self.individuals) - 1
            last_gen = self.individuals[last_gen_id]
            last_gen_history = self.historical_fitness[last_gen_id]
            adapter = PipelineAdapter()
            for individual, ind_fitness in zip(last_gen, last_gen_history):
                ind_path = Path(save_dir, str(last_gen_id), str(individual.uid))
                additional_info = \
                    {'fitness_name': self._objective.metric_names,
                     'fitness_value': ind_fitness}
                PipelineTemplate(adapter.restore(individual)).\
                    export_pipeline(path=ind_path, additional_info=additional_info, datetime_in_path=False)
        except Exception as ex:
            self._log.exception(ex)

    def save(self, json_file_path: Union[str, os.PathLike] = None) -> Optional[str]:
        if json_file_path is None:
            return json.dumps(self, indent=4, cls=Serializer)
        with open(json_file_path, mode='w') as json_file:
            json.dump(self, json_file, indent=4, cls=Serializer)

    @staticmethod
    def load(json_str_or_file_path: Union[str, os.PathLike] = None) -> 'OptHistory':
        def load_as_file_path():
            with open(json_str_or_file_path, mode='r') as json_file:
                return json.load(json_file, cls=Serializer)

        def load_as_json_str():
            return json.loads(json_str_or_file_path, cls=Serializer)

        if isinstance(json_str_or_file_path, os.PathLike):
            return load_as_file_path()

        try:
            return load_as_json_str()
        except json.JSONDecodeError:
            return load_as_file_path()

    @staticmethod
    def clean_results(dir_path: Optional[str] = None):
        """Clearn the directory tree with previously dumped history results."""
        if dir_path and os.path.isdir(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
            os.mkdir(dir_path)

    @property
    def historical_fitness(self) -> Sequence[Sequence[Union[float, Sequence[float]]]]:
        """Return sequence of histories of generations per each metric"""
        if self._objective.is_multi_objective:
            historical_fitness = []
            num_metrics = len(self._objective.metrics)
            for objective_num in range(num_metrics):
                # history of specific objective for each generation
                objective_history = [[ind.fitness.values[objective_num] for ind in generation]
                                     for generation in self.individuals]
                historical_fitness.append(objective_history)
        else:
            historical_fitness = [[pipeline.fitness.value for pipeline in pop] for pop in self.individuals]
        return historical_fitness

    @property
    def all_historical_fitness(self):
        historical_fitness = self.historical_fitness
        if self._objective.is_multi_objective:
            all_historical_fitness = []
            for obj_num in range(len(historical_fitness)):
                all_historical_fitness.append(list(itertools.chain(*historical_fitness[obj_num])))
        else:
            all_historical_fitness = list(itertools.chain(*historical_fitness))
        return all_historical_fitness

    @property
    def all_historical_quality(self):
        if self._objective.is_multi_objective:
            metric_position = get_metric_position(self._objective.metrics, QualityMetricsEnum)
            all_historical_quality = self.all_historical_fitness[metric_position]
        else:
            all_historical_quality = self.all_historical_fitness
        return all_historical_quality

    @property
    def historical_pipelines(self):
        adapter = PipelineAdapter()
        return [
            PipelineTemplate(adapter.restore(ind))
            for ind in list(itertools.chain(*self.individuals))
        ]

    @property
    def show(self):
        return OptHistoryVisualizer(self)

    def get_leaderboard(self, top_n: int = 10) -> str:
        """
        Prints ordered description of the best solutions in history
        :param top_n: number of solutions to print
        """
        # Take only the first graph's appearance in history
        individuals_with_positions \
            = list({ind.graph.descriptive_id: (ind, gen_num, ind_num)
                    for gen_num, gen in enumerate(self.individuals)
                    for ind_num, ind in reversed(list(enumerate(gen)))}.values())

        top_individuals = sorted(individuals_with_positions,
                                 key=lambda pos_ind: pos_ind[0].fitness, reverse=True)[:top_n]

        output = io.StringIO()
        separator = ' | '
        header = separator.join(['Position', 'Fitness', 'Generation', 'Pipeline'])
        print(header, file=output)
        for ind_num, ind_with_position in enumerate(top_individuals):
            individual, gen_num, ind_num = ind_with_position
            positional_id = f'g{gen_num}-i{ind_num}'
            print(separator.join([f'{ind_num:>3}, '
                                  f'{str(individual.fitness):>8}, '
                                  f'{positional_id:>8}, '
                                  f'{individual.graph.descriptive_id}']), file=output)

        # add info about initial assumptions (stored as zero generation)
        for i, individual in enumerate(self.individuals[0]):
            ind = f'I{i}'
            positional_id = '-'
            print(separator.join([f'{ind:>3}'
                                  f'{str(individual.fitness):>8}',
                                  f'{positional_id}',
                                  f'{individual.graph.descriptive_id}']), file=output)
        return output.getvalue()


def log_to_history(population: PopulationT,
                   generations: GenerationKeeper,
                   history: OptHistory,
                   save_dir: Optional[os.PathLike] = None):
    """
    Default variant of callback that preserves optimisation history
    :param history: OptHistory for logging
    :param population: list of individuals obtained in last iteration
    :param generations: keeper of the best individuals from all iterations
    :param save_dir: directory for saving history to. None if saving to a file is not required.
    """
    history.add_to_history(population)
    history.add_to_archive_history(generations.best_individuals)
    if save_dir:
        history.save_current_results(save_dir)
