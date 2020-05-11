from enum import Enum
from core.repository.model_types_repository import ModelTypesIdsEnum


class BenchmarkModelTypesEnum(Enum):
    h2o = ModelTypesIdsEnum.h2o,
    tpot = ModelTypesIdsEnum.tpot,
    autokeras = 'AutoKeras',
    mlbox = 'mlbox',
    fedot = 'fedot',
    baseline = 'baseline'
