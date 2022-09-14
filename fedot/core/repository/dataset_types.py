from fedot.core.utilities.data_structures import ComparableEnum as Enum


class DataTypesEnum(Enum):
    """An enumeration

    Args:
        table: table with columns as features for predictions, by default == ``feature_table``
        ts: one dimensional array - time series, by default == ``time_series``
        multi_ts: table with different variant of time-series for the same variable as columns
           (used for extending train sample), by default == ``multiple_time_series``
        text: table, where cells contains text, by default == ``text``
        image: images represented as 3d arrays, by default == ``image``
    """

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
