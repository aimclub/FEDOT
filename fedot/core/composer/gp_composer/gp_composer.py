import csv
import os
from datetime import datetime
from typing import Collection, Optional, Sequence, Tuple, Union
from uuid import uuid4

from golem.core.log import is_test_session
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimizer import GraphOptimizer

from fedot.core.caching.operations_cache import OperationsCache
from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_objective_eval import (
    PipelineObjectiveEvaluate,
)
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import (
    PipelineComposerRequirements,
)
from fedot.core.utils import default_fedot_data_dir


class GPComposer(Composer):
    """
    Genetic programming based composer

    :param optimizer: optimizer generated in ComposerBuilder.
    :param composer_requirements: requirements for composition process.
    :param operations_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    :param predictions_cache: Cache manager for fit/predict node's predictions, optional.
    """

    def __init__(self, optimizer: GraphOptimizer,
                 composer_requirements: PipelineComposerRequirements,
                 operations_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None,
                 predictions_cache: Optional[PredictionsCache] = None):
        super().__init__(optimizer, composer_requirements)
        self.composer_requirements = composer_requirements
        self.operations_cache: Optional[OperationsCache] = operations_cache
        self.preprocessing_cache: Optional[PreprocessingCache] = preprocessing_cache
        self.predictions_cache: Optional[PredictionsCache] = predictions_cache

        self.best_models: Collection[Pipeline] = ()

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, Sequence[Pipeline]]:
        # Define data source
        data_splitter = DataSourceSplitter(self.composer_requirements.cv_folds,
                                           shuffle=True)
        data_producer = data_splitter.build(data)

        parallelization_mode = self.composer_requirements.parallelization_mode
        if parallelization_mode == 'populational':
            n_jobs_for_evaluation = 1
        elif parallelization_mode == 'sequential':
            n_jobs_for_evaluation = self.composer_requirements.n_jobs
        else:
            raise ValueError(f'Unknown parallelization_mode: {parallelization_mode}')

        # Define objective function
        objective_evaluator = PipelineObjectiveEvaluate(objective=self.optimizer.objective,
                                                        data_producer=data_producer,
                                                        time_constraint=self.composer_requirements.max_graph_fit_time,
                                                        operations_cache=self.operations_cache,
                                                        preprocessing_cache=self.preprocessing_cache,
                                                        predictions_cache=self.predictions_cache,
                                                        validation_blocks=data_splitter.validation_blocks,
                                                        eval_n_jobs=n_jobs_for_evaluation)
        objective_function = objective_evaluator.evaluate

        # Define callback for computing intermediate metrics if needed
        if self.composer_requirements.collect_intermediate_metric:
            self.optimizer.set_evaluation_callback(objective_evaluator.evaluate_intermediate_metrics)

        # Finally, run optimization process
        opt_result = self.optimizer.optimise(objective_function)

        best_model, self.best_models = self._convert_opt_results_to_pipeline(opt_result)
        self.log.info('GP composition finished')

        if is_test_session() and self.predictions_cache is not None:
            self._save_predictions_cache()

        return best_model

    def _convert_opt_results_to_pipeline(self, opt_result: Sequence[OptGraph]) -> Tuple[Optional[Pipeline],
                                                                                        Sequence[Pipeline]]:
        adapter = self.optimizer.graph_generation_params.adapter
        multi_objective = self.optimizer.objective.is_multi_objective
        best_pipelines = [adapter.restore(graph) for graph in opt_result]
        if not best_pipelines:
            return None, []
        chosen_best_pipeline = best_pipelines if multi_objective else best_pipelines[0]
        return chosen_best_pipeline, best_pipelines

    def _save_predictions_cache(self):
        """
        Saves predictions cache effectiveness and usage statistics to a CSV file.
        """
        if not self.predictions_cache._db.use_stats:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        uuid = uuid4()
        directory = os.path.join(f"{str(default_fedot_data_dir())}", "saved_cache_effectiveness")
        os.makedirs(directory, exist_ok=True)

        predictions_file_path = os.path.join(directory, f"{timestamp}_predictions_{str(uuid)}.csv")

        # Write predictions cache data
        with open(predictions_file_path, "w", newline="") as f:
            # Write effectiveness ratio
            writer = csv.DictWriter(f, self.predictions_cache.effectiveness_ratio.keys())
            writer.writeheader()
            writer.writerow(self.predictions_cache.effectiveness_ratio)

            # Write usage statistics
            stats_writer = csv.writer(f)
            stats_writer.writerows(self.predictions_cache._db.retrieve_stats())
