from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.random.random_search import RandomSearchOptimizer

from fedot.core.composer.composer import Composer
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline


class RandomSearchComposer(Composer):
    def __init__(self, optimizer: RandomSearchOptimizer):
        super().__init__(optimizer=optimizer)

    def compose_pipeline(self, data: InputData) -> Pipeline:
        train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

        def prepared_objective(pipeline: Pipeline) -> Fitness:
            pipeline.fit(train_data)
            return self.optimizer.objective(pipeline, reference_data=test_data)

        best_pipeline = self.optimizer.optimise(prepared_objective)[0]
        return best_pipeline
