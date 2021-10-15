from fedot.core.utils import ComparableEnum as Enum
from fedot.core.utils import SerializableEnumMeta
from fedot.shared import EnumSerializer


class DataTypesEnum(EnumSerializer, Enum, metaclass=SerializableEnumMeta):
    # Table with columns as features for predictions
    table = 'feature_table'

    # One dimensional array - time series
    ts = 'time_series'

    # Table, where cells contains text
    text = 'text'

    # Images represented as 3d arrays
    image = 'image'
