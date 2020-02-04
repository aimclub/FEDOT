from enum import Enum


class DataTypesEnum(Enum):
    pass


class NumericalDataTypesEnum(DataTypesEnum):
    vector = "numerical_vector"
    table = "numerical_table"
    ts = "numerical_timeseries"


class CategorialDataTypesEnum(DataTypesEnum):
    vector = "categorial_vector"
    table = "categorial_table"
    ts = "categorial_timeseries"


class SpecialDataTypesEnum(DataTypesEnum):
    text = "text"
    binary = "binary"
