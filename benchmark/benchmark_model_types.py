from enum import Enum


class BenchmarkModelTypesEnum(Enum):
    h2o = 'h2o',
    tpot = 'tpot',
    autokeras = 'AutoKeras',
    mlbox = 'mlbox',
    fedot = 'fedot',
    baseline = 'baseline'
