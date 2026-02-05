import datetime
from copy import deepcopy
from datetime import timedelta
from functools import partial
from typing import Optional, Tuple, Union, Sequence

import optuna
from dask.distributed import wait
from distributed import Client, LocalCluster
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune
from optuna import Trial, Study
from optuna.trial import FrozenTrial


class DaskOptunaTuner(BaseTuner):
    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 objectives_number: int = 1):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds,
                         timeout,
                         n_jobs,
                         deviation)
        self.objectives_number = objectives_number
        self.study = None
        self.iterations = 100
        self.n_trials = 10

    def _dask_backend_tune(self, predefined_objective, show_progress):
        self.storage = optuna.integration.DaskStorage()
        # self.storage = optuna.integration.dask.DaskStorage()
        self.study = optuna.create_study(storage=self.storage,
                                         direction='minimize')  # ['minimize'] * self.objectives_number
        # Submit self.n_trials different optimization tasks, where each task runs self.iterations optimization trials
        tuning_cluster_params = dict(processes=False, n_workers=1, threads_per_worker=4, memory_limit='auto')
        cluster = LocalCluster(**tuning_cluster_params)
        client = Client(cluster)
        futures = [client.submit(self.study.optimize,
                                 predefined_objective,
                                 n_trials=self.iterations,
                                 n_jobs=self.n_jobs,
                                 timeout=self.timeout.seconds,
                                 callbacks=[self.early_stopping_callback],
                                 show_progress_bar=show_progress) for _ in range(self.n_trials)]
        wait(futures)
        print(f"Best params: {self.study.best_params}")

    def tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> \
            Union[DomainGraphForTune, Sequence[DomainGraphForTune]]:
        graph = self.adapter.adapt(graph)
        predefined_objective = partial(self.objective, graph=graph)
        is_multi_objective = self.objectives_number > 1
        self.init_check(graph)
        init_parameters, has_parameters_to_optimize = self._get_initial_point(graph)

        if not has_parameters_to_optimize:
            self._stop_tuning_with_message(f'Graph {graph.graph_description} has no parameters to optimize')
            tuned_graphs = self.init_graph
        else:
            # Enqueue initial point to try
            verbosity_level = optuna.logging.INFO if show_progress else optuna.logging.WARNING
            optuna.logging.set_verbosity(verbosity_level)
            self._dask_backend_tune(predefined_objective, show_progress)
            tuned_graphs = self.set_arg_graph(graph, self.study.best_trials[0].params) if not is_multi_objective else \
                [self.set_arg_graph(deepcopy(graph), best_trial.params) for best_trial in self.study.best_trials]
            self.was_tuned = True

        final_graphs = self.final_check(tuned_graphs, is_multi_objective)
        final_graphs = self.adapter.restore(final_graphs)
        return final_graphs

    def objective(self, trial: Trial, graph: OptGraph) -> Union[float, Sequence[float,]]:
        new_parameters = self._get_parameters_from_trial(graph, trial)
        new_graph = BaseTuner.set_arg_graph(graph, new_parameters)
        metric_value = self.get_metric_value(new_graph)
        return metric_value

    def _get_parameters_from_trial(self, graph: OptGraph, trial: Trial) -> dict:
        new_parameters = {}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name, {})

            for parameter_name, parameter_properties in tunable_node_params.items():
                node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

                parameter_type = parameter_properties.get('type')
                sampling_scope = parameter_properties.get('sampling-scope')
                if parameter_type == 'discrete':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_int(node_op_parameter_name, *sampling_scope)})
                elif parameter_type == 'continuous':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_float(node_op_parameter_name, *sampling_scope)})
                elif parameter_type == 'categorical':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_categorical(node_op_parameter_name, *sampling_scope)})
        return new_parameters

    def _get_initial_point(self, graph: OptGraph) -> Tuple[dict, bool]:
        initial_parameters = {}
        has_parameters_to_optimize = False
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name)

            if tunable_node_params:
                has_parameters_to_optimize = True
                tunable_initial_params = {get_node_operation_parameter_label(node_id, operation_name, p):
                                          node.parameters[p] for p in node.parameters if p in tunable_node_params}
                if tunable_initial_params:
                    initial_parameters.update(tunable_initial_params)
        return initial_parameters, has_parameters_to_optimize

    def early_stopping_callback(self, study: Study, trial: FrozenTrial):
        if self.early_stopping_rounds is not None:
            current_trial_number = trial.number
            best_trial_number = study.best_trial.number
            should_stop = (current_trial_number - best_trial_number) >= self.early_stopping_rounds
            if should_stop:
                self.log.debug('Early stopping rounds criteria was reached')
                study.stop()


def tune_pipeline_industrial(self, train_data: InputData, pipeline_gp_composed: Pipeline) -> Pipeline:
    """ Launch tuning procedure for obtained pipeline by composer """
    timeout_for_tuning = abs(self.timer.determine_resources_for_tuning()) / 60
    tuner = (TunerBuilder(self.params.task)
             .with_tuner(OptunaTuner)  # DaskOptunaTuner
             .with_metric(self.metrics[0])
             .with_iterations(DEFAULT_TUNING_ITERATIONS_NUMBER)
             .with_timeout(datetime.timedelta(minutes=timeout_for_tuning))
             .with_eval_time_constraint(self.params.composer_requirements.max_graph_fit_time)
             .with_requirements(self.params.composer_requirements)
             .build(train_data))

    if self.timer.have_time_for_tuning():
        # Tune all nodes in the pipeline
        with self.timer.launch_tuning():
            self.was_tuned = False
            self.log.message(f'Hyperparameters tuning started with {round(timeout_for_tuning)} min. timeout')
            tuned_pipeline = tuner.tune(pipeline_gp_composed)
            self.log.message('Hyperparameters tuning finished')
    else:
        self.log.message(f'Time for pipeline composing was {str(self.timer.composing_spend_time)}.\n'
                         f'The remaining {max(0, round(timeout_for_tuning, 1))} seconds are not enough '
                         f'to tune the hyperparameters.')
        self.log.message('Composed pipeline returned without tuning.')
        tuned_pipeline = pipeline_gp_composed
    self.was_tuned = tuner.was_tuned
    return tuned_pipeline
