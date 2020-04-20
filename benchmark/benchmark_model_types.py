from enum import Enum


class ModelTypesEnum(Enum):
    h2o = 'H2O',
    tpot = 'TPOT',
    autokeras = 'AutoKeras',
    mlbox = 'MLBox',
    fedot = 'FEDOT',
    baseline = 'baseline'
