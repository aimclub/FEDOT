from fedot.core.utilities.data_structures import ComparableEnum as Enum


class DataTypesEnum(Enum):
    # Table with columns as features for predictions
    table = 'feature_table'

    # One dimensional array - time series
    ts = 'time_series'

    # Table with different variant of time-series for the same variable as columns (used for extending train sample)
    multi_ts = 'multiple_time_series'

    # Table, where cells contains text
    text = 'text'

    # Images represented as 3d arrays
    image = 'image'
