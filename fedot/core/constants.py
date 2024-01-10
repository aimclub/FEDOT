from fedot.core.repository.tasks import TaskTypesEnum

MINIMAL_SECONDS_FOR_TUNING = 15
DEFAULT_TUNING_ITERATIONS_NUMBER = 100000
DEFAULT_API_TIMEOUT_MINUTES = 5.0
DEFAULT_FORECAST_LENGTH = 30
COMPOSING_TUNING_PROPORTION = 0.6

MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION = 100
MIN_NUMBER_OF_GENERATIONS = 3

FRACTION_OF_UNIQUE_VALUES = 0.95

default_data_split_ratio_by_task = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}
