import csv
import itertools
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import (Any, List)

from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.utils.population_utils import get_metric_position
from fedot.core.repository.quality_metrics_repository import QualityMetricsEnum
from fedot.core.utils import default_fedot_data_dir


@dataclass
class ParentOperator:
    operator_name: str
    operator_type: str
    parent_objects: List[PipelineTemplate]


class OptHistory:
    """
    Contain history, convert Pipeline to PipelineTemplate, save history to csv
    """

    def __init__(self, metrics=None):
        self.metrics = metrics
        self.individuals = []
        self.archive_history = []
        self.pipelines_comp_time_history = []
        self.archive_comp_time_history = []
        self.parent_operators = []

    def _convert_pipeline_to_template(self, pipeline):
        pipeline_template = PipelineTemplate(pipeline)
        return pipeline_template

    def add_to_history(self, individuals: List[Any]):
        new_individuals = []
        pipelines_comp_time = []
        parent_operators = []
        try:
            for ind in individuals:
                pipeline = ind.graph  # was restored outside
                new_ind = deepcopy(ind)
                new_ind.graph = self._convert_pipeline_to_template(pipeline)
                new_individuals.append(new_ind)
                if hasattr(pipeline, 'computation_time'):
                    pipelines_comp_time.append(pipeline.computation_time)
                else:
                    pipelines_comp_time.append(-1)
                parent_operators.append(ind.parent_operators)
            self.individuals.append(new_individuals)
            self.pipelines_comp_time_history.append(pipelines_comp_time)
        except Exception as ex:
            print(f'Cannot add to history: {ex}')

        self.parent_operators.append(parent_operators)

    def add_to_archive_history(self, individuals: List[Any]):
        try:
            new_individuals = []
            archive_comp_time = []
            for ind in individuals:
                new_ind = deepcopy(ind)
                new_ind.graph = self._convert_pipeline_to_template(ind.graph)
                new_individuals.append(new_ind)
                archive_comp_time.append(ind.graph.computation_time)
            self.archive_history.append(new_individuals)
            self.archive_comp_time_history.append(archive_comp_time)
        except Exception as ex:
            print(f'Cannot add to archive history: {ex}')

    def write_composer_history_to_csv(self, file='history.csv'):
        history_dir = os.path.join(default_fedot_data_dir(), 'composing_history')
        file = os.path.join(history_dir, file)
        if not os.path.isdir(history_dir):
            os.mkdir(history_dir)
        self._write_header_to_csv(file)
        idx = 0
        for gen_num, gen_inds in enumerate(self.individuals):
            for ind_num, ind in enumerate(gen_inds):
                if self.is_multi_objective:
                    fitness = ind.fitness.values
                else:
                    fitness = ind.fitness
                row = [idx, gen_num, fitness, len(ind.graph.operation_templates), ind.graph.depth,
                       self.pipelines_comp_time_history[gen_num][ind_num]]
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

    @property
    def historical_fitness(self):
        if self.is_multi_objective:
            historical_fitness = []
            for objective_num in range(len(self.individuals[0][0].fitness.values)):
                objective_history = [[pipeline.fitness.values[objective_num] for pipeline in pop] for pop in self.individuals]
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
        return [ind.graph for ind in list(itertools.chain(*self.individuals))]

    @property
    def is_multi_objective(self):
        return type(self.individuals[0][0].fitness) is MultiObjFitness
