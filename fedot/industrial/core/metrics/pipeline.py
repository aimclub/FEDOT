import os
import pickle
import traceback
from pathlib import Path
from typing import Callable, Iterable, Tuple
from uuid import uuid4

import numpy as np
from fedot.core.data.input_data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.objective.objective import to_fitness

from fedot.industrial.tools.serialisation.path_lib import PROJECT_PATH

DataSource = Callable[[], Iterable[Tuple[InputData, InputData]]]


def save_pipeline_for_debug(pipeline: Pipeline, train_data: InputData,
                            test_data: InputData, exception: Exception, stack_trace: str):
    try:
        tmp_folder = Path(PROJECT_PATH, 'debug_info')
        if not tmp_folder.exists():
            os.mkdir(tmp_folder)

        pipeline_id = str(uuid4())
        base_path = Path(tmp_folder, pipeline_id)
        pipeline.save(f'{base_path}/pipeline', is_datetime_in_path=False)

        with open(f'{base_path}/train_data.pkl', 'wb') as file:
            pickle.dump(train_data, file)
        with open(f'{base_path}/test_data.pkl', 'wb') as file:
            pickle.dump(test_data, file)
        with open(f'{base_path}/exception.txt', 'w') as file:
            print(exception, file=file)
            print(stack_trace, file=file)
    except Exception as ex:
        print(ex)


def industrial_evaluate_pipeline(self, graph: Pipeline) -> Fitness:
    # Seems like a workaround for situation when logger is lost
    #  when adapting and restoring it to/from OptGraph.
    try:
        graph.log = self._log
    except Exception:
        _ = 1

    graph_id = graph.root_node.descriptive_id
    self._log.debug(f'Pipeline {graph_id} fit started')

    folds_metrics = []
    folds_list = list(enumerate(self._data_producer()))
    val_blocks = self._validation_blocks
    for fold_id, (train_data, test_data) in folds_list:
        try:
            prepared_pipeline = self.prepare_graph(graph, train_data, fold_id, self._eval_n_jobs)
        except Exception as ex:
            self._log.warning(f'Unsuccessful pipeline fit during fitness evaluation. '
                              f'Skipping the pipeline. Exception <{ex}> on {graph_id}')
            stack_trace = traceback.format_exc()
            prepared_pipeline = self.prepare_graph(graph, train_data, fold_id, self._eval_n_jobs)
            save_pipeline_for_debug(graph, train_data, test_data, ex, stack_trace)
            break  # if even one fold fails, the evaluation stops
        evaluated_fitness = self._objective(prepared_pipeline,
                                            reference_data=test_data,
                                            validation_blocks=val_blocks)
        if evaluated_fitness.valid:
            folds_metrics.append(evaluated_fitness.values)
        else:
            self._log.warning(f'Invalid fitness after objective evaluation. Skipping the graph: {graph_id}',
                              raise_if_test=False)
        if self._do_unfit:
            graph.unfit()
    if folds_metrics:
        folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
        self._log.debug(f'Pipeline {graph_id} with evaluated metrics: {folds_metrics}')
    else:
        folds_metrics = None

    # prepared_pipeline.

    return to_fitness(folds_metrics, self._objective.is_multi_objective)
