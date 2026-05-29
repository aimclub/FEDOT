from typing import Any, List, Protocol, Optional, Callable, Union, Tuple, Sequence, TYPE_CHECKING
import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum
from golem.core.optimisers.fitness import Fitness
from golem.core.tuning.tuner_interface import BaseTuner

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.data.multi_modal import MultiModalData
    from fedot.core.data.merge.data_merger import DataMerger

class EvaluatorProtocol(Protocol):
    def evaluate(self, graph: 'Pipeline') -> Fitness:
        ...


class SearchSpaceProtocol(Protocol):
    def get_parameters_dict(self) -> dict:
        ...


class DefaultMutationsProtocol(Protocol):
    @staticmethod
    def __call__(task_type: TaskTypesEnum, params: Any) -> Sequence[Any]:
        ...


class DataMergerProtocol(Protocol):
    @staticmethod
    def get(outputs: List[OutputData]) -> 'DataMerger':
        ...

    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        ...

    @staticmethod
    def find_main_output(outputs: List[OutputData]) -> OutputData:
        ...

    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        ...

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        ...


class ImageMergerProtocol(Protocol):
    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        ...

    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        ...


class TSMergerProtocol(Protocol):
    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        ...

    def merge_targets(self, targets: List[np.ndarray]) -> np.ndarray:
        ...

    def preprocess_predicts(self, predicts: List[np.ndarray]) -> List[np.ndarray]:
        ...

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        ...


class TextMergerProtocol(Protocol):
    def merge_predicts(self, predicts: List[np.ndarray]) -> np.ndarray:
        ...

    def postprocess_predicts(self, merged: np.ndarray) -> np.ndarray:
        ...


class DataSourceSplitterProtocol(Protocol):
    def build(self, data: Union[InputData, 'MultiModalData']) -> Callable:
        ...


class SplitterProtocol(Protocol):
    def split_any(self, data: InputData, split_ratio: float, shuffle: bool,
                  stratify: bool, random_seed: int, **kwargs) -> Tuple[InputData, InputData]:
        ...

    def split_time_series(self, data: InputData, validation_blocks: Optional[int] = None,
                          **kwargs) -> Tuple[InputData, InputData]:
        ...


class OperationPredictProtocol(Protocol):
    def predict(self, fitted_operation, data: InputData,
                params: Optional[Any] = None, output_mode: str = 'default') -> OutputData:
        ...

    def predict_for_fit(self, fitted_operation, data: InputData,
                        params: Optional[Any] = None, output_mode: str = 'default') -> OutputData:
        ...

    def _predict(self, fitted_operation, data: InputData, params: Optional[Any] = None,
                 output_mode: str = 'default', is_fit_stage: bool = False,
                 predictions_cache: Optional[Any] = None, fold_id: Optional[int] = None,
                 descriptive_id: Optional[str] = None) -> OutputData:
        ...


class LaggedTransformerProtocol(Protocol):
    def _update_column_types(self, output_data: OutputData) -> None:
        ...

    def transform(self, input_data: InputData) -> OutputData:
        ...

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        ...

    def _check_and_correct_window_size(self, time_series: np.ndarray, forecast_length: int) -> None:
        ...


class TopologicalFeaturesProtocol(Protocol):
    def fit(self, input_data: InputData) -> Any:
        ...

    def transform(self, input_data: InputData) -> np.ndarray:
        ...


class TsSmoothingProtocol(Protocol):
    def transform(self, input_data: InputData) -> OutputData:
        ...


class TunerClassProtocol(Protocol):
    def __call__(self, objective_evaluate: Any, task: Any, iterations: int,
                 max_lead_time: Optional[float] = None, **kwargs) -> BaseTuner:
        ...


class ApiComposerTuneProtocol(Protocol):
    def __call__(self, train_data: InputData, pipeline: 'Pipeline',
                 execution_plan: Optional[Any] = None) -> 'Pipeline':
        ...


class ReproductionProtocol(Protocol):
    def reproduce(self, population: List[Any], evaluator: Any, **kwargs) -> List[Any]:
        ...

    def reproduce_uncontrolled(self, population: List[Any], **kwargs) -> List[Any]:
        ...


class VerificationRulesProtocol(Protocol):
    class_rules: List[Callable]
    ts_rules: List[Callable]
    common_rules: List[Callable]