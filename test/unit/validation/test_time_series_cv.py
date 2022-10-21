import datetime
import logging

from examples.advanced.time_series_forecasting.composing_pipelines import get_available_operations
from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.validation.split import ts_cv_generator
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline, get_ts_data

log = default_log(prefix=__name__)


def configure_experiment():
    """ Generates a time series of 100 elements. The prediction is performed
    for five elements ahead
    """
    # Default number of validation blocks
    validation_blocks = 3
    forecast_len = 5

    time_series, _ = get_ts_data(n_steps=105, forecast_length=forecast_len)

    return forecast_len, validation_blocks, time_series


def test_ts_cv_generator_correct():
    """ Checks if the split into training and test for time series cross
    validation is correct

    By default, the number of validation blocks for each fold is five
    """
    folds = 2
    forecast_len, validation_blocks, time_series = configure_experiment()
    ts_len = len(time_series.idx)

    # The "in-sample validation" is carried out for each fold
    validation_elements_per_fold = forecast_len * validation_blocks
    # Entire length of validation for all folds
    validation_horizon = validation_elements_per_fold * folds

    i = 0
    for train_data, test_data in ts_cv_generator(time_series, folds, validation_blocks, log):
        train_len = len(train_data.idx)
        assert train_len == ts_len - validation_horizon
        validation_horizon -= validation_elements_per_fold
        i += 1
    assert i == folds


def test_cv_folds_too_large_correct():
    """ Checks whether cases where the number of folds is too large, causing
    the number of elements to be validated to be greater than the number of elements
    in the time series itself, are adequately handled

    In this case a hold-out validation with 1 fold and 3 validation blocks must be performed
    """
    folds = 50
    forecast_len, validation_blocks, time_series = configure_experiment()

    i = 0
    for train_data, test_data in ts_cv_generator(time_series, folds, validation_blocks, log):
        i += 1
        assert len(train_data.idx) == 85
    assert i == 1


def test_tuner_cv_correct():
    """
    Checks if the tuner works correctly when using cross validation for
    time series
    """
    folds = 2
    forecast_len, validation_blocks, time_series = configure_experiment()

    simple_pipeline = get_simple_ts_pipeline()
    tuner = TunerBuilder(time_series.task)\
        .with_tuner(PipelineTuner)\
        .with_metric(RegressionMetricsEnum.MAE)\
        .with_cv_folds(folds) \
        .with_validation_blocks(validation_blocks)\
        .with_iterations(1) \
        .with_timeout(datetime.timedelta(minutes=1))\
        .build(time_series)
    _ = tuner.tune(simple_pipeline)
    is_tune_succeeded = True
    assert is_tune_succeeded


def test_composer_cv_correct():
    """ Checks if the composer works correctly when using cross validation for
    time series """
    forecast_len, validation_blocks, time_series = configure_experiment()

    primary_operations, secondary_operations = get_available_operations()

    # Composer parameters
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations,
        num_of_generations=2,
        timeout=datetime.timedelta(seconds=5),
        cv_folds=2,
        validation_blocks=validation_blocks,
        show_progress=False)
    parameters = GPGraphOptimizerParameters(
        pop_size=2,
        crossover_prob=0.8,
        mutation_prob=0.8,
    )

    init_pipeline = get_simple_ts_pipeline()
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    builder = ComposerBuilder(task=time_series.task). \
        with_optimizer_params(parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipelines([init_pipeline])
    composer = builder.build()

    obtained_pipeline = composer.compose_pipeline(data=time_series)
    assert isinstance(obtained_pipeline, Pipeline)


def test_api_cv_correct():
    """ Checks if the composer works correctly when using cross validation for
    time series through api """
    folds = 2
    forecast_len, validation_blocks, time_series = configure_experiment()
    timeout = 0.05
    composer_params = {'max_depth': 2,
                       'max_arity': 2,
                       'preset': 'fast_train',
                       'cv_folds': folds,
                       'num_of_generations': 1,
                       'validation_blocks': validation_blocks,
                       'show_progress': False}
    task_parameters = TsForecastingParams(forecast_length=forecast_len)

    model = Fedot(problem='ts_forecasting',
                  timeout=timeout,
                  task_params=task_parameters,
                  logging_level=logging.DEBUG,
                  **composer_params)
    fedot_model = model.fit(features=time_series)
    assert fedot_model is not None
