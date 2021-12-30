import datetime

from sklearn.metrics import mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import get_available_operations
from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import \
    PipelineComposerRequirements
from fedot.core.log import default_log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.validation.split import ts_cv_generator
from fedot.core.validation.tune.time_series import cv_time_series_predictions
from test.unit.tasks.test_forecasting import get_simple_ts_pipeline, get_ts_data

log = default_log(__name__)


def configure_experiment():
    """ Generates a time series of 100 elements. The prediction is performed
    for five elements ahead
    """
    # Default number of validation blocks
    validation_blocks = 3
    forecast_len = 5

    time_series, _ = get_ts_data(n_steps=105, forecast_length=forecast_len)
    log = default_log(__name__)

    return log, forecast_len, validation_blocks, time_series


def test_ts_cv_generator_correct():
    """ Checks if the split into training and test for time series cross
    validation is correct

    By default, the number of validation blocks for each fold is three
    """
    folds = 2
    log, forecast_len, validation_blocks, time_series = configure_experiment()
    ts_len = len(time_series.idx)

    # The "in-sample validation" is carried out for each fold
    validation_elements_per_fold = forecast_len * validation_blocks
    # Entire length of validation for all folds
    validation_horizon = validation_elements_per_fold * folds

    i = 0
    for train_data, test_data, vb_number in ts_cv_generator(time_series, folds, validation_blocks, log):
        train_len = len(train_data.idx)
        assert train_len == ts_len - validation_horizon
        validation_horizon -= validation_elements_per_fold
        i += 1
    assert i == folds


def test_cv_folds_too_large_correct():
    """ Checks whether cases where the number of folds is too large, causing
    the number of elements to be validated to be greater than the number of elements
    in the time series itself, are adequately handled

    In this case a hold-out validation must be performed
    """
    folds = 50
    log, forecast_len, validation_blocks, time_series = configure_experiment()

    i = 0
    for train_data, test_data, vb_number in ts_cv_generator(time_series, folds, validation_blocks, log):
        i += 1
        assert len(train_data.idx) == 95
    assert i == 1


def test_tuner_cv_correct():
    """
    Checks if the tuner works correctly when using cross validation for
    time series
    """
    folds = 2
    _, forecast_len, validation_blocks, time_series = configure_experiment()

    simple_pipeline = get_simple_ts_pipeline()
    tuned = simple_pipeline.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                                loss_params=None,
                                                input_data=time_series,
                                                iterations=1, timeout=1,
                                                cv_folds=folds,
                                                validation_blocks=validation_blocks)
    is_tune_succeeded = True
    assert is_tune_succeeded


def test_cv_ts_predictions_correct():
    folds_len_list = []
    for folds in range(2, 4):
        _, forecast_len, validation_blocks, time_series = configure_experiment()

        simple_pipeline = get_simple_ts_pipeline()
        predictions, target = cv_time_series_predictions(reference_data=time_series,
                                                         pipeline=simple_pipeline,
                                                         log=log,
                                                         cv_folds=folds,
                                                         validation_blocks=validation_blocks)
        folds_len_list.append(len(predictions))
    assert folds_len_list[0] < folds_len_list[1]


def test_composer_cv_correct():
    """ Checks if the composer works correctly when using cross validation for
    time series """
    folds = 2
    _, forecast_len, validation_blocks, time_series = configure_experiment()

    primary_operations, secondary_operations = get_available_operations()

    # Composer parameters
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=3, pop_size=2, num_of_generations=2,
        crossover_prob=0.8, mutation_prob=0.8,
        timeout=datetime.timedelta(seconds=5),
        cv_folds=folds,
        validation_blocks=validation_blocks)

    init_pipeline = get_simple_ts_pipeline()
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    builder = ComposerBuilder(task=time_series.task). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipelines([init_pipeline])
    composer = builder.build()

    obtained_pipeline = composer.compose_pipeline(data=time_series, is_visualise=False)
    assert isinstance(obtained_pipeline, Pipeline)


def test_api_cv_correct():
    """ Checks if the composer works correctly when using cross validation for
    time series through api """
    folds = 2
    _, forecast_len, validation_blocks, time_series = configure_experiment()
    composer_params = {'max_depth': 1,
                       'max_arity': 2,
                       'timeout': 0.05,
                       'preset': 'fast_train',
                       'cv_folds': folds,
                       'validation_blocks': validation_blocks}
    task_parameters = TsForecastingParams(forecast_length=forecast_len)

    model = Fedot(problem='ts_forecasting',
                  composer_params=composer_params,
                  task_params=task_parameters,
                  verbose_level=2)
    fedot_model = model.fit(features=time_series)
    is_succeeded = True
    assert is_succeeded
