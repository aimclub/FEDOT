from abc import ABC, abstractmethod
from typing import Iterable, Any, Dict, Optional
from enum import Enum
import torch
from typing import Literal, Union
from transformers import PreTrainedModel
from torch.nn import Module
from functools import reduce

from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.utils.checkpoint_registration_rules import (
    build_checkpoint_registration_request,
    execute_checkpoint_registration,
)
from fedot.industrial.core.models.nn.utils.hook_runtime_rules import (
    build_hook_runtime_payload,
    execute_stage_hooks,
)
from fedot.industrial.core.models.nn.utils.runtime_metadata_rules import (
    OutputCompatibilityContext,
    RegistryCheckpointContext,
    build_output_compatibility_context,
    build_registry_checkpoint_context,
)
from fedot.industrial.core.models.nn.utils.hooks_collection import HooksCollection
from fedot.industrial.core.models.nn.utils.interfaces import (ITrainer, IHookable)
from fedot.industrial.core.repository.constanst_repository import ModelLearningHooks,LoggingHooks,TorchLossesConstant

HookType = Literal['start', 'end']


class BaseTrainer(ITrainer, IHookable):

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}
        self.learning_params = self.params.get('custom_learning_params', {})

        self._hooks = []
        self._additional_hooks = []
        self.hooks_collection = HooksCollection()
        self.hooks = self.hooks_collection

        self.trainer_objects = {
            'optimizer': None,
            'scheduler': None,
            'trainer': None,
            'stop': False,
        }
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        self.model: Union['PreTrainedModel', 'Module', None] = self.params.get("model", None)
        self.device = default_device()

    def register_additional_hooks(self, hooks: Iterable[Enum]) -> None:
        for hook in hooks:
            if hook not in self._additional_hooks:
                self._additional_hooks.append(hook)

    def _init_hooks(self) -> None:
        raise NotImplementedError("Subclasses must implement _init_hooks")

    def execute_hooks(self, hook_type: HookType, epoch: int, **kwargs) -> None:
        payload = build_hook_runtime_payload(
            trainer_objects=self.trainer_objects,
            history=self.history,
            extra=kwargs,
        )
        execute_stage_hooks(self.hooks_collection, hook_type, epoch, payload)

    @abstractmethod
    def fit(self, input_data: Any, supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        pass

    @abstractmethod
    def predict(self, input_data: Any, output_mode: str = "default") -> Any:
        pass

    # def save_model(self, path: str) -> None:
    #     if self.model is not None:
    #         if hasattr(self.model, 'save_pretrained'):
    #             self.model.save_pretrained(path)
    #         else:
    #             torch.save(self.model.state_dict(), path)
    #         print(f"Model saved to {path}")
    #     else:
    #         print("No model to save")

    # def load_model(self, path: str) -> None:
    #     """Load the model - default implementation"""
    #     if os.path.exists(path):
    #         if self.model is None:
    #             print("Model not initialized, cannot load weights")
    #             return

    #         if hasattr(self.model, 'from_pretrained'):
    #             self.model = self.model.from_pretrained(path)
    #         else:
    #             state_dict = torch.load(path, map_location=self.device)
    #             self.model.load_state_dict(state_dict)
    #         print(f"Model loaded from {path}")
    #     else:
    #         print(f"Model path {path} does not exist")

    # def _clear_cache(self):
    #     """Clear CUDA cache - shared by BaseNeuralModel and LLMTrainer"""
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()

    def _clear_cache(self):
        """Clear GPU cache using ModelRegistry cleanup methods."""
        try:
            from fedot.industrial.tools.registry import ModelRegistry
            registry = ModelRegistry()
            registry.force_cleanup()
        except Exception:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _compute_loss(self, criterion, model_output, target, stage='train', epoch=None):
        if hasattr(model_output, 'loss'):
            quality_loss = model_output.loss
        else:
            quality_loss = criterion(model_output, target)
        if isinstance(model_output, torch.Tensor):
            additional_losses = {name: coef * criterion(model_output, target)
                                 for name, (criterion, coef) in self.custom_criterions.items()
                                 if hasattr(TorchLossesConstant, name)}
            # additional_losses.update({name: coef * criterion(self.model)
            #                           for name, (criterion, coef) in self.custom_criterions.items()
            #                           if hasattr(StructureCriterions, name)})
            for name, val in additional_losses.items():
                self.history[f'{stage}_{name}_loss'].append((epoch, val))
        final_loss = reduce(torch.add, additional_losses.values(), quality_loss)
        return final_loss

    @property
    def is_quantised(self) -> bool:
        return getattr(self.model, '_is_quantised', False)

    @property
    def optimizer(self) -> Any:
        return self.trainer_objects.get('optimizer')

    @optimizer.setter
    def optimizer(self, value: Any) -> None:
        self.trainer_objects['optimizer'] = value

    @property
    def scheduler(self) -> Any:
        return self.trainer_objects.get('scheduler')

    @scheduler.setter
    def scheduler(self, value: Any) -> None:
        self.trainer_objects['scheduler'] = value

    def _normalize_kwargs(self, kwargs: Dict[str, Any], allowed_keys: set) -> Dict[str, Any]:
        normalized = {}
        synonym_mapping = {
            'num_epochs': 'num_train_epochs',
            'epochs': 'num_train_epochs',
            'batch_size': 'per_device_train_batch_size',
            'train_batch_size': 'per_device_train_batch_size',
            'eval_batch_size': 'per_device_eval_batch_size',
            'learning_rate': 'learning_rate',
            'lr': 'learning_rate',
        }

        for key, value in kwargs.items():
            if key in allowed_keys:
                normalized[key] = value
            elif key in synonym_mapping and synonym_mapping[key] in allowed_keys:
                normalized[synonym_mapping[key]] = value

        return normalized

    def _extract_output_fields(self, input_data: Any) -> OutputCompatibilityContext:
        """Extract common fields from input_data for CompressionOutputData creation.

        Uses a mapping approach similar to _normalize_kwargs for consistent field extraction.

        Args:
            input_data: InputData or CompressionInputData instance

        Returns:
            Typed compatibility context with features and loader metadata.
        """
        return build_output_compatibility_context(input_data)

    def _register_model_checkpoint(self, model: Any, fedcore_id: str = None,
                                   stage: Optional[str] = None) -> RegistryCheckpointContext:
        """Register model in ModelRegistry and return checkpoint information.

        Args:
            model: Model to register
            fedcore_id: FedCore instance identifier (optional)
            stage: Stage name (e.g., 'before', 'after') (optional)

        Returns:
            Typed checkpoint context with model/checkpoint identifiers.
        """
        try:
            from fedot.industrial.tools.registry.model_registry import (ModelRegistry)

            registry = ModelRegistry()
            request = build_checkpoint_registration_request(
                model_present=model is not None,
                stage=stage,
                explicit_fedcore_id=fedcore_id,
                trainer_fedcore_id=getattr(self, '_fedcore_id', None),
                thread_local_context=registry.get_registry_context(),
            )
            return execute_checkpoint_registration(
                request=request,
                model=model,
                register_model=registry.register_model,
                get_checkpoint_path=registry.get_checkpoint_path,
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"ModelRegistry registration failed: {e}")
            return build_registry_checkpoint_context(fedcore_id=fedcore_id)
