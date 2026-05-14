from typing import Dict, Any, Optional, Callable
from fedot.extensions.registry import get_registered_extension, register_extension
from fedot.core.context.industrial_manifest import FEDOT_INDUSTRIAL_MANIFEST

register_extension(FEDOT_INDUSTRIAL_MANIFEST)

class ExecutionContext:
    def __init__(self, extension_name: str = "core", extra_params: Optional[Dict[str, Any]] = None):
        self.extension_name = extension_name
        self.extra_params = extra_params or {}
        self._instances: Dict[str, Any] = {}
        self._overridden: Dict[str, Any] = {}

        self._manifest = None
        if extension_name != "core":
            from fedot.extensions.registry import _REGISTERED_EXTENSIONS
            manifest = _REGISTERED_EXTENSIONS.get(extension_name)
            if manifest is None:
                raise ValueError(f"Extension '{extension_name}' not registered")
            self._manifest = manifest

        self._core_implementations = self._get_core_implementations()

        self._protocol_classes = self._core_implementations.copy()
        if self._manifest and self._manifest.protocols:
            self._protocol_classes.update(self._manifest.protocols)

    def _get_core_implementations(self) -> Dict[str, Callable]:
        from fedot.core.context.default_backend import (
            CoreSplitter, CoreDataMerger, CoreImageMerger,
            CoreTSMerger, CoreTextMerger, CoreTuner,
            CoreDataSourceSplitter, CoreOperationPredict,
            CoreLaggedTransformer, CoreTopologicalFeatures,
            CoreTsSmoothing, CoreApiComposerTune, CoreReproduction,
            CoreSearchSpace, CoreDefaultMutations, CoreEvaluator
        )
        return {
            "splitter": CoreSplitter,
            "data_merger": CoreDataMerger,
            "image_merger": CoreImageMerger,
            "ts_merger": CoreTSMerger,
            "text_merger": CoreTextMerger,
            "tuner_class": CoreTuner,
            "data_source_splitter": CoreDataSourceSplitter,
            "operation_predict": CoreOperationPredict,
            "lagged_transformer": CoreLaggedTransformer,
            "topological_features": CoreTopologicalFeatures,
            "ts_smoothing": CoreTsSmoothing,
            "api_composer_tune": CoreApiComposerTune,
            "reproduction": CoreReproduction,
            "search_space": CoreSearchSpace,
            "default_mutations": CoreDefaultMutations,
            "evaluator": CoreEvaluator,
        }

    def _get_protocol_class(self, protocol_name: str) -> Callable:
        if self._manifest and self._manifest.protocols:
            if protocol_name in self._manifest.protocols:
                return self._manifest.protocols[protocol_name]

        if protocol_name in self._core_implementations:
            return self._core_implementations[protocol_name]

        raise ValueError(f"No implementation for protocol '{protocol_name}'")

    def _get_instance(self, protocol_name: str) -> Any:
        if protocol_name not in self._instances:
            protocol_class = self._get_protocol_class(protocol_name)
            self._instances[protocol_name] = protocol_class(**self.extra_params)
        return self._instances[protocol_name]

    @property
    def splitter(self):
        return self._get_instance("splitter")

    @property
    def data_merger(self):
        return self._get_instance("data_merger")

    @property
    def image_merger(self):
        return self._get_instance("image_merger")

    @property
    def ts_merger(self):
        return self._get_instance("ts_merger")

    @property
    def text_merger(self):
        return self._get_instance("text_merger")

    @property
    def tuner_class(self):
        return self._get_instance("tuner_class")

    @property
    def data_source_splitter(self):
        return self._get_instance("data_source_splitter")

    @property
    def operation_predict(self):
        return self._get_instance("operation_predict")

    @property
    def lagged_transformer(self):
        return self._get_instance("lagged_transformer")

    @property
    def topological_features(self):
        return self._get_instance("topological_features")

    @property
    def ts_smoothing(self):
        return self._get_instance("ts_smoothing")

    @property
    def api_composer_tune(self):
        return self._get_instance("api_composer_tune")

    @property
    def reproduction(self):
        return self._get_instance("reproduction")

    @property
    def search_space(self):
        return self._get_instance("search_space")

    @property
    def default_mutations(self):
        return self._get_instance("default_mutations")

    @property
    def evaluator(self):
        return self._get_instance("evaluator")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ('extra_params', '_instances', '_overridden', '_protocol_classes',
                    '_manifest', '_core_implementations', 'extension_name'):
            super().__setattr__(name, value)
        else:
            self._overridden[name] = value

    def __getattr__(self, name: str):
        if name in self._overridden:
            return self._overridden[name]

        if name in ('_protocol_classes', '_instances', '_manifest', '_core_implementations'):
            return super().__getattribute__(name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")