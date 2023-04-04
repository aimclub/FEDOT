from typing import Optional

from statsforecast.models import HistoricAverage, Naive, RandomWalkWithDrift, WindowAverage, SeasonalWindowAverage

from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsforecasting import \
    StatsForecastingImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class HistoricAverageImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = HistoricAverage(**params.to_dict())


class RepeatLastImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = Naive(**params.to_dict())


class RandomWalkWithDriftImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = RandomWalkWithDrift(**params.to_dict())


class WindowAverageImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = WindowAverage(**params.to_dict())


class SeasonalWindowAverageImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = SeasonalWindowAverage(**params.to_dict())