import sys 
import datetime
from fedot.core.composer.metrics import QualityMetric
from fedot.confidence_intervals.metrics import quantile_loss
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.tuning.simultaneous import SimultaneousTuner
from fedot.core.data.data import InputData, OutputData

def quantile_loss_tuners(up_quantile, low_quantile,task,train_input,
                       validations_blocks = 2,n_jobs = -1,show_progress = True):
    class Quantile_Loss_low(QualityMetric):
        default_value = sys.maxsize
        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            value = quantile_loss(reference.target, predicted.predict,quantile = low_quantile)
            return value
    class Quantile_Loss_up(QualityMetric):
        default_value = sys.maxsize
        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            value = quantile_loss(reference.target, predicted.predict,quantile = up_quantile)
            return value       
        
    composer_requirements = PipelineComposerRequirements()
    composer_requirements.validation_blocks = validations_blocks
    composer_requirements.n_jobs= n_jobs
    composer_requirements.show_progress = show_progress 
                    
    low_tuner = TunerBuilder(task = task) \
                 .with_tuner(SimultaneousTuner) \
                 .with_metric(Quantile_Loss_low.get_value) \
                 .with_iterations(10) \
                 .with_timeout(datetime.timedelta(minutes=1)) \
                 .with_requirements(composer_requirements) \
                 .build(train_input)
    up_tuner = TunerBuilder(task = task) \
                 .with_tuner(SimultaneousTuner) \
                 .with_metric(Quantile_Loss_up.get_value) \
                 .with_iterations(10) \
                 .with_timeout(datetime.timedelta(minutes=1)) \
                 .with_requirements(composer_requirements) \
                 .build(train_input)
    return {'low_tuner':low_tuner, 'up_tuner':up_tuner}