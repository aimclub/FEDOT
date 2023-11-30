from golem.core.tuning.iopt_tuner import IOptTuner

from fedot.core.composer.metrics import MSE
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def tune_pipeline(pipeline: Pipeline,
                  train_data: InputData,
                  test_data: InputData,
                  tuner_iter_num: int = 100):

    task = Task(TaskTypesEnum.regression)
    requirements = PipelineComposerRequirements(cv_folds=3, n_jobs=-1)
    metric = RegressionMetricsEnum.MSE

    # Fit initial pipeline
    pipeline.fit(train_data)
    before_tuning_predicted = pipeline.predict(test_data)
    # Obtain test metric before tuning
    metric_before_tuning = MSE().metric(test_data, before_tuning_predicted)

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(IOptTuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_iterations(tuner_iter_num) \
        .with_additional_params(eps=0.02, r=1.5, refine_solution=True) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

    # Fit tuned pipeline
    tuned_pipeline.fit(train_data)
    after_tuning_predicted = tuned_pipeline.predict(test_data)
    # Obtain test metric after tuning
    metric_after_tuning = MSE().metric(test_data, after_tuning_predicted)

    print(f'\nMetric before tuning: {metric_before_tuning}')
    print(f'Metric after tuning: {metric_after_tuning}')
    return tuned_pipeline


if __name__ == '__main__':
    pipeline = (PipelineBuilder()
                .add_node('dtreg', 0)
                .add_node('knnreg', 1)
                .join_branches('rfr')
                .build())
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

    data = InputData.from_csv(data_path,
                              task=Task(TaskTypesEnum.regression))
    train_data, test_data = train_test_data_setup(data)
    tuned_pipeline = tune_pipeline(pipeline, train_data, test_data, tuner_iter_num=200)
