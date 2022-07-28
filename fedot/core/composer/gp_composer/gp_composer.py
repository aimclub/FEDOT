from typing import Collection, Optional, Sequence, Tuple, Union

from fedot.core.caching.pipelines_cache import OperationsCache
from fedot.core.caching.preprocessing_cache import PreprocessingCache
from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.optimisers.optimizer import GraphOptimizer
from fedot.core.pipelines.pipeline import Pipeline


class GPComposer(Composer):
    """
    Genetic programming based composer
    :param optimiser: optimiser generated in ComposerBuilder.
    :param composer_requirements: requirements for composition process.
    :param pipelines_cache: Cache manager for fitted models, optional.
    :param preprocessing_cache: Cache manager for optional preprocessing encoders and imputers, optional.
    :param initial_pipelines: defines the initial state of the population.
    """

    def __init__(self, optimiser: GraphOptimizer,
                 composer_requirements: PipelineComposerRequirements,
                 initial_pipelines: Optional[Sequence[Pipeline]] = None,
                 history: Optional[OptHistory] = None,
                 pipelines_cache: Optional[OperationsCache] = None,
                 preprocessing_cache: Optional[PreprocessingCache] = None):

        super().__init__(optimiser, composer_requirements, initial_pipelines)
        self.composer_requirements = composer_requirements

        self.optimiser: GraphOptimizer = optimiser
        self.pipelines_cache: Optional[OperationsCache] = pipelines_cache
        self.preprocessing_cache: Optional[PreprocessingCache] = preprocessing_cache
        self.history: Optional[OptHistory] = history
        self.best_models: Collection[Pipeline] = ()

        self.objective_builder = DataObjectiveBuilder(optimiser.objective,
                                                      composer_requirements.max_pipeline_fit_time,
                                                      composer_requirements.cv_folds,
                                                      composer_requirements.validation_blocks,
                                                      pipelines_cache, preprocessing_cache)

    def compose_pipeline(self, data: Union[InputData, MultiModalData]) -> Union[Pipeline, Sequence[Pipeline]]:
        # shuffle data if necessary
        data.shuffle()

        # Keep history of optimization
        if self.history:
            self.history.clean_results()

        # Define objective function
        validation_blocks = self.composer_requirements.validation_blocks
        objective_evaluator = self.objective_builder.build(data, validation_blocks=validation_blocks)
        objective_function = objective_evaluator.evaluate

        # Define callback for computing intermediate metrics if needed
        if self.composer_requirements.collect_intermediate_metric:
            self.optimiser.set_evaluation_callback(objective_evaluator.evaluate_intermediate_metrics)

        # Finally, run optimization process
        opt_result = self.optimiser.optimise(objective_function,
                                             show_progress=self.composer_requirements.show_progress)

        best_model, self.best_models = self._convert_opt_results_to_pipeline(opt_result)
        self.log.info('GP composition finished')
        return best_model

    def _convert_opt_results_to_pipeline(self, opt_result: Sequence[OptGraph]) -> Tuple[Pipeline, Sequence[Pipeline]]:
        adapter = self.optimiser.graph_generation_params.adapter
        multi_objective = self.optimiser.objective.is_multi_objective
        best_pipelines = [adapter.restore(graph) for graph in opt_result]
        chosen_best_pipeline = best_pipelines if multi_objective else best_pipelines[0]
        return chosen_best_pipeline, best_pipelines

    @staticmethod
    def tune_pipeline(pipeline: Pipeline, data: InputData, time_limit):
        raise NotImplementedError()
