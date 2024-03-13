from fedot.core.repository.tasks import TaskTypesEnum

MINIMAL_SECONDS_FOR_TUNING = 15
"""Minimal seconds for tuning."""

DEFAULT_TUNING_ITERATIONS_NUMBER = 100000
"""Default number of tuning iterations."""

DEFAULT_API_TIMEOUT_MINUTES = 5.0
"""Default API timeout in minutes."""

DEFAULT_FORECAST_LENGTH = 30
"""Default forecast length."""

COMPOSING_TUNING_PROPORTION = 0.6
"""Proportion of data used for composing tuning."""

BEST_QUALITY_PRESET_NAME = 'best_quality'
"""Name of the preset for best quality."""

FAST_TRAIN_PRESET_NAME = 'fast_train'
"""Name of the preset for fast training."""

AUTO_PRESET_NAME = 'auto'
"""Name of the preset for auto tuning."""

MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION = 100
"""Minimal number of pipelines for evaluation."""

MIN_NUMBER_OF_GENERATIONS = 3
"""Minimum number of generations."""

FRACTION_OF_UNIQUE_VALUES = 0.95
"""Fraction of unique values."""

default_data_split_ratio_by_task = {
    TaskTypesEnum.classification: 0.8,
    TaskTypesEnum.regression: 0.8,
    TaskTypesEnum.ts_forecasting: 0.5
}
"""Default data split ratio by task."""

PCA_MIN_THRESHOLD_TS = 7
"""Minimum threshold for PCA in TS forecasting."""
