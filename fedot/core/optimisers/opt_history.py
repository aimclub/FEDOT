import csv
import datetime
import itertools
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from uuid import uuid4

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.serializers import Serializer

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.individual import Individual

from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.utils.population_utils import get_metric_position
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum
from fedot.core.utils import default_fedot_data_dir


@dataclass
class ParentOperator:
    operator_name: str
    operator_type: str
    parent_objects: List['Individual']
    uid: str = None

    def __post_init__(self):
        if not self.uid:
            self.uid = str(uuid4())


class OptHistory:
    """
    Contain history, convert Pipeline to PipelineTemplate, save history to csv
    """

    def __init__(self, metrics: List[Callable[..., float]] = None, save_folder=None):
        self.metrics = metrics
        self.individuals: List[List['Individual']] = []
        self.archive_history: List[List['Individual']] = []
        self.save_folder: str = save_folder if save_folder \
            else f'composing_history_{datetime.datetime.now().timestamp()}'

    def add_to_history(self, individuals: List['Individual']):
        self.individuals.append([deepcopy(ind) for ind in individuals])

    def add_to_archive_history(self, individuals: List['Individual']):
        self.archive_history.append([ind for ind in individuals])

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = self._get_save_path()
        file = os.path.join(history_dir, file)
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self._write_header_to_csv(file)
        idx = 0
        adapter = PipelineAdapter()
        for gen_num, gen_inds in enumerate(self.individuals):
            for ind_num, ind in enumerate(gen_inds):
                if self.is_multi_objective:
                    fitness = ind.fitness.values
                else:
                    fitness = ind.fitness
                ind_pipeline_template = adapter.restore_as_template(ind.graph, ind.computation_time)
                row = [
                    idx, gen_num, fitness,
                    len(ind_pipeline_template.operation_templates), ind_pipeline_template.depth, ind.computation_time
                ]
                self._add_history_to_csv(file, row)
                idx += 1

    def _write_header_to_csv(self, f):
        with open(f, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            metric_str = 'metric'
            if self.is_multi_objective:
                metric_str += 's'
            row = ['index', 'generation', metric_str, 'quantity_of_operations', 'depth', 'computation_time']
            writer.writerow(row)

    def _add_history_to_csv(self, f: str, row: List[Any]):
        with open(f, 'a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(row)

    def save_current_results(self, path: Optional[str] = None):
        if not path:
            path = self._get_save_path()
        try:
            last_gen_id = len(self.individuals) - 1
            last_gen = self.individuals[last_gen_id]
            for ind_id, individual in enumerate(last_gen):
                # TODO support multi-objective case
                ind_path = os.path.join(path, str(last_gen_id), str(individual.graph.uid))
                additional_info = \
                    {'fitness_name': self.short_metrics_names[0],
                     'fitness_value': self.historical_fitness[last_gen_id][ind_id]}
                PipelineAdapter().restore_as_template(
                    individual.graph, individual.computation_time
                ).export_pipeline(path=ind_path, additional_info=additional_info, datetime_in_path=False)
        except Exception as ex:
            print(ex)

    def save(self, json_file_path: os.PathLike = None) -> Optional[str]:
        if json_file_path is None:
            return json.dumps(self, indent=4, cls=Serializer)
        with open(json_file_path, mode='w') as json_fp:
            json.dump(self, json_fp, indent=4, cls=Serializer)

    @staticmethod
    def load(json_str_or_file_path: Union[str, os.PathLike] = None) -> 'OptHistory':
        try:
            return json.loads(json_str_or_file_path, cls=Serializer)
        except json.JSONDecodeError as exc:
            with open(json_str_or_file_path, mode='r') as json_fp:
                return json.load(json_fp, cls=Serializer)

    def clean_results(self, path: Optional[str] = None):
        if not path:
            path = os.path.join(default_fedot_data_dir(), self.save_folder)
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)

    @property
    def short_metrics_names(self):
        # TODO refactor
        possible_short_names = ['RMSE', 'MSE', 'ROCAUC', 'MAE']
        short_names = []
        for full_name in self.metrics:
            is_found = False
            for candidate_short_name in possible_short_names:
                if candidate_short_name in str(full_name):
                    short_names.append(candidate_short_name)
                    is_found = True
                    break
            if not is_found:
                short_names.append(str(full_name))

        return short_names

    @property
    def historical_fitness(self):
        if self.is_multi_objective:
            historical_fitness = []
            for objective_num in range(len(self.individuals[0][0].fitness.values)):
                objective_history = [[pipeline.fitness.values[objective_num] for pipeline in pop] for pop in
                                     self.individuals]
                historical_fitness.append(objective_history)
        else:
            historical_fitness = [[pipeline.fitness for pipeline in pop] for pop in self.individuals]
        return historical_fitness

    @property
    def all_historical_fitness(self):
        historical_fitness = self.historical_fitness
        if self.is_multi_objective:
            all_historical_fitness = []
            for obj_num in range(len(historical_fitness)):
                all_historical_fitness.append(list(itertools.chain(*historical_fitness[obj_num])))
        else:
            all_historical_fitness = list(itertools.chain(*historical_fitness))
        return all_historical_fitness

    @property
    def all_historical_quality(self):
        if self.is_multi_objective:
            if self.metrics:
                metric_position = get_metric_position(self.metrics, QualityMetricsEnum)
            else:
                metric_position = 0
            all_historical_quality = self.all_historical_fitness[metric_position]
        else:
            all_historical_quality = self.all_historical_fitness
        return all_historical_quality

    @property
    def historical_pipelines(self):
        adapter = PipelineAdapter()
        return [
            adapter.restore_as_template(ind.graph, ind.computation_time)
            for ind in list(itertools.chain(*self.individuals))
        ]

    @property
    def is_multi_objective(self):
        return type(self.individuals[0][0].fitness) is MultiObjFitness

    def _get_save_path(self):
        if os.path.sep in self.save_folder:
            # Defined path is full - there is no need to use default dir
            # Create folder if it's not exists
            if os.path.isdir(self.save_folder) is False:
                os.makedirs(self.save_folder)
            return self.save_folder
        else:
            return os.path.join(default_fedot_data_dir(), self.save_folder)
