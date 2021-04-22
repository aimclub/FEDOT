from fedot.core.utils import ComparableEnum as Enum


class DataTypesEnum(Enum):
    table = 'feature_table'

    # 2d dataset with timeseries as target and external variables as features
    ts = 'time_series'

    # 2d dataset with time series forecasted by model
    forecasted_ts = 'time_series_forecasted'

    # 2d dataset with lagged features - (n, window_len * features)
    ts_lagged_table = 'time_series_lagged_table'

    text = 'text'
