from enum import Enum


class DataTypesEnum(Enum):
    pass


class NumericalDataTypesEnum(DataTypesEnum):
    vector = 'numerical_vector'
    table = 'numerical_table'
    ts = 'numerical_timeseries'


class CategoricalDataTypesEnum(DataTypesEnum):
    vector = 'categoriсal_vector'
    table = 'categoriсal_table'
    ts = 'categoriсal_timeseries'


class SpecialDataTypesEnum(DataTypesEnum):
    text = 'text'
    binary = 'binary'
