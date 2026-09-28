from golem.utilities.data_structures import ComparableEnum as Enum


class DataTypesEnum(Enum):
    """Dataset type taxonomy used across FEDOT.

    Canonical transition targets:
        - ``tabular`` for table-like datasets and text after embedding/encoding
        - ``ts`` for time-series-like tensor paths and image-like tensor layouts

    Legacy aliases are intentionally preserved because a large part of FEDOT still
    references ``table``, ``multi_ts``, ``text``, and ``image`` directly.
    ``TensorData`` and new tensor-aware paths normalize those values through a
    compatibility mapper instead of forcing an immediate repo-wide rewrite.
    """

    tabular = 'table'
    table = 'table'

    ts = 'time_series'
    multi_ts = 'multi_time_series'

    text = 'text'
    image = 'image'
