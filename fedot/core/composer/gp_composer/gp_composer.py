from typing import Collection, Optional, Sequence, Tuple, Union

from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimizer import GraphOptimizer

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements


class GPComposer(Composer):
    """
    Genetic programming based composer

    :param optimizer: optimizer generated in ComposerBuilder.
    :param composer_requirements: requirements for composition process.
    :param pipelines_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    """

    def __init__(self, optimizer: GraphOptimizer,
                 composer_requirements: PipelineComposerRequirements,
                 pipelines_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None):
        super().__init__(optimizer, composer_requirements)
        self.composer_requirements = composer_requirements
        self.pipelines_cache: Optional[OperationsCache] = pipelines_cache
        self.preprocessing_cache: Optional[PreprocessingCache] = preprocessing_cache

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
                                                        pipelines_cache=self.pipelines_cache,
                                                        preprocessing_cache=self.preprocessing_cache,
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
