from golem.utilities.data_structures import ComparableEnum as Enum


class DataTypesEnum(Enum):
    """An enumeration

    Args:
        tabular: table with columns as features for predictions, by default == ``table``
        ts: one dimensional and multivariate time series, by default == ``time_series``
    """

    ts = 'time_series'  # any time series and images(like in industrial)
    tabular = 'table'
