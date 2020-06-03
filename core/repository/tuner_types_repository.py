from enum import Enum


class TunerTypeEnum(Enum):
    pass


class SklearnTunerTypeEnum(TunerTypeEnum):
    grid = 'grid',
    bayes = 'bayes',
    rand = 'rand'
    crand = 'crand'


class CustomTunerTypeEnum(TunerTypeEnum):
    custom = 'custom'
