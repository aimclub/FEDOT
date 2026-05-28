from typing import Dict, Any, Optional, Callable
from fedot.extensions.registry import get_registered_extension, register_extension


class ExecutionContext:
    def __init__(self, extension_name: str = "industrial", extra_params: Optional[Dict[str, Any]] = None):
        self.extension_name = extension_name
        self.extra_params = extra_params or {}
        self._instances: Dict[str, Any] = {}
        self._overridden: Dict[str, Any] = {}

        manifest = get_registered_extension(extension_name)
        if manifest is None:
            raise ValueError(f"Extension '{extension_name}' not registered")
        self._manifest = manifest

        self._protocol_classes = self._manifest.protocols or {}

    def _get_protocol_class(self, protocol_name: str) -> Callable:
        if protocol_name in self._protocol_classes:
            return self._protocol_classes[protocol_name]
        raise ValueError(f"No implementation for protocol '{protocol_name}'")

    def _get_instance(self, protocol_name: str) -> Any:
        if protocol_name in self._overridden:
            override = self._overridden[protocol_name]
            if isinstance(override, type):
                return override(**self.extra_params)
            return override

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
                    '_manifest', 'extension_name'):
            super().__setattr__(name, value)
        else:
            self._overridden[name] = value

    def __getattr__(self, name: str):
        if name in self._overridden:
            return self._overridden[name]

        if name in ('_protocol_classes', '_instances', '_manifest'):
            return super().__getattribute__(name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
