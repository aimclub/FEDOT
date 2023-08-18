import sys
import datetime

from golem.core.tuning.simultaneous import SimultaneousTuner
from fedot.core.composer.metrics import QualityMetric
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import Task

from fedot.core.pipelines.prediction_intervals.metrics import quantile_loss


def quantile_loss_tuners(up_quantile: float,
                         low_quantile: float,
                         task: Task,
                         train_input: InputData,
                         show_progress: bool,
                         validation_blocks: int,
                         n_jobs: int,
                         iterations: int,
                         minutes: float):
    """This function builds default tuners for quantile loss method.

    Args:
        up_quantile: upper quantile
        low_quantile: low quantile
        task: task specifying horizon for tuning
        train_input: train data for tuners
        show_progress: whether to show progress during tuning
        validation_blocks: number of validation blocks for tuners
        n_jobs: n_jobs for tuners
        iterations: number iterations for tuners
        minutes: number minutes for tuners.

    Returns:
        a dictionary consisitng of upper and low SimultaneousTuner tuners.
        """
    class Quantile_Loss_low(QualityMetric):
        default_value = sys.maxsize

        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            value = quantile_loss(reference.target, predicted.predict, quantile=low_quantile)
            return value

    class Quantile_Loss_up(QualityMetric):
        default_value = sys.maxsize

        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            value = quantile_loss(reference.target, predicted.predict, quantile=up_quantile)
            return value

    composer_requirements = PipelineComposerRequirements()
    composer_requirements.validation_blocks = validation_blocks
    composer_requirements.n_jobs = n_jobs
    composer_requirements.show_progress = show_progress

    low_tuner = (TunerBuilder(task=task)
                 .with_tuner(SimultaneousTuner)
                 .with_metric(Quantile_Loss_low.get_value)
                 .with_iterations(iterations)
                 .with_timeout(datetime.timedelta(minutes=minutes))
                 .with_requirements(composer_requirements)
                 .build(train_input))
    up_tuner = (TunerBuilder(task=task)
                .with_tuner(SimultaneousTuner)
                .with_metric(Quantile_Loss_up.get_value)
                .with_iterations(iterations)
                .with_timeout(datetime.timedelta(minutes=minutes))
                .with_requirements(composer_requirements)
                .build(train_input))

    return {'low_tuner': low_tuner, 'up_tuner': up_tuner}
