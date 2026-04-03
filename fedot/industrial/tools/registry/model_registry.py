import os
import gc
import logging
from typing import Optional, Tuple, Dict, Any
from threading import Lock, local

import torch

try:
    from fedot.core.pipelines.pipeline import Pipeline
except ImportError:
    Pipeline = None

from .checkpoint_manager import CheckpointManager
from .registry_storage import RegistryStorage
from .metrics_tracker import MetricsTracker

_registry_context = local()

_STAGE_NORMALIZATION = {
    'before': ['before', 'initial'],
    'after': ['after']
}

_MODEL_ATTRS_TO_CLEAN = ['model_before', '_model_before_cached', 'model_after',
                         '_model_after_cached', 'model', 'model_for_inference']

_CLEANUP_ITERATIONS = 3


class ModelRegistry:
    """Thread-safe Singleton model registry for FedCore pipeline."""

    _instance = None
    _initialized = False
    _lock = Lock()

    def __new__(cls, auto_cleanup: bool = True):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, auto_cleanup: bool = True):
        if not ModelRegistry._initialized:
            with ModelRegistry._lock:
                if not ModelRegistry._initialized:
                    base_dir = os.environ.get("FEDCORE_MODEL_REGISTRY_PATH", "llm_output")
                    self.checkpoint_manager = CheckpointManager(base_dir, auto_cleanup=auto_cleanup)
                    self.storage = RegistryStorage(base_dir)
                    self.metrics_tracker = MetricsTracker()
                    self.auto_cleanup = auto_cleanup
                    self.logger = logging.getLogger(self.__class__.__name__)
                    self.logger.setLevel(logging.INFO)
                    ModelRegistry._initialized = True
                    self.logger.info(f"ModelRegistry initialized: auto_cleanup={auto_cleanup}, base_dir={base_dir}")

    @staticmethod
    def set_registry_context(fedcore_id: str, model_id: str) -> None:
        _registry_context.fedcore_id = fedcore_id
        _registry_context.model_id = model_id

    @staticmethod
    def get_registry_context() -> Optional[Tuple[str, str]]:
        fedcore_id = getattr(_registry_context, 'fedcore_id', None)
        model_id = getattr(_registry_context, 'model_id', None)
        return (fedcore_id, model_id) if (fedcore_id and model_id) else None

    @staticmethod
    def clear_registry_context() -> None:
        for attr in ('fedcore_id', 'model_id'):
            if hasattr(_registry_context, attr):
                delattr(_registry_context, attr)

    def _normalize_stage(self, stage: Optional[str]) -> Optional[str]:
        if stage is None:
            return None
        stage_lower = stage.lower()
        for normalized, keywords in _STAGE_NORMALIZATION.items():
            if any(kw in stage_lower for kw in keywords):
                return normalized
        return "after"

    def _inherit_mode(self, fedcore_id: str, model_id: str, mode: Optional[str]) -> Optional[str]:
        if mode is not None:
            return mode
        if existing := self.storage.get_latest_record(fedcore_id, model_id):
            if inherited_mode := existing.get('mode'):
                self.logger.info(f"Inherited mode={inherited_mode} from previous record for model_id={model_id}")
                return inherited_mode
        return None

    def _log_memory_stats(self, context: str) -> Optional[Dict[str, float]]:
        if not (self.auto_cleanup and torch.cuda.is_available()):
            return None
        stats = self.checkpoint_manager.get_gpu_memory_stats()
        self.logger.info(f"GPU memory {context}: {stats.get('allocated_gb', 0):.4f} GB")
        return stats

    def _create_record(self, fedcore_id: str, model_id: str, version: str,
                       checkpoint_path: str, model_path: Optional[str] = None,
                       stage: Optional[str] = None, mode: Optional[str] = None) -> Dict[str, Any]:
        return {
            "record": str(self.metrics_tracker.generate_model_id()),
            "fedcore": fedcore_id,
            "model": model_id,
            "created_at": version,
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "stage": stage,
            "mode": mode,
            "metrics": {},
        }

    def _resolve_stage_mode(self, fedcore_id: str, model_id: str, stage: Optional[str], mode: Optional[str]) -> Tuple[
        Optional[str], Optional[str]]:
        stage = self._normalize_stage(stage)
        mode = self._inherit_mode(fedcore_id, model_id, mode)
        return stage, mode

    def _save_checkpoint_and_record(self, fedcore_id: str, model_id: str, model=None,
                                    model_path: Optional[str] = None,
                                    delete_model_after_save: bool = True,
                                    stage: Optional[str] = None, mode: Optional[str] = None) -> str:
        version = self.metrics_tracker.generate_version()
        safe_timestamp = self.metrics_tracker.sanitize_timestamp(version)

        self._log_memory_stats("before saving")
        checkpoint_bytes = self.checkpoint_manager.serialize_to_bytes(model, model_path)

        checkpoint_path = (model_path if model_path and os.path.isfile(model_path) else
                           self.checkpoint_manager.generate_checkpoint_path(fedcore_id, model_id, safe_timestamp))

        if checkpoint_path != model_path:
            self.checkpoint_manager.save_to_file(checkpoint_bytes, checkpoint_path)

        record = self._create_record(fedcore_id, model_id, version, checkpoint_path,
                                     model_path, stage, mode)
        self.storage.append_record(fedcore_id, record)

        if delete_model_after_save and model is not None:
            self.logger.info(f"Deleting model {model_id} from memory")
            self._delete_model_from_memory(model)
            self.logger.info("Model deleted from memory")

        self._log_memory_stats("after cleanup")
        return model_id

    def register_model(self, fedcore_id: str, model=None, model_path: str = None,
                       pipeline_params: dict = None, note: str = "initial", params_format: str = 'yaml',
                       delete_model_after_save: bool = True, stage: Optional[str] = None,
                       mode: Optional[str] = None) -> str:
        self.logger.info(
            f"register_model called: fedcore_id={fedcore_id}, model_type={type(model).__name__ if model else 'None'}, stage={stage}, mode={mode}")

        model_id = self.metrics_tracker.generate_model_id(model, model_path)
        stage, mode = self._resolve_stage_mode(fedcore_id, model_id, stage, mode)

        return self._save_checkpoint_and_record(fedcore_id, model_id, model, model_path,
                                                delete_model_after_save, stage, mode)

    def register_changes(self, fedcore_id: str, model_id: str, model=None,
                         pipeline_params: dict = None, note: str = "update", params_format: str = 'yaml',
                         delete_model_after_save: bool = True, stage: Optional[str] = None,
                         mode: Optional[str] = None):
        self.logger.info(
            f"register_changes called: fedcore_id={fedcore_id}, model_id={model_id}, stage={stage}, mode={mode}")

        existing = self.storage.get_latest_record(fedcore_id, model_id)
        if existing is None:
            self.logger.warning("No existing record found, calling register_model instead")
            self.register_model(fedcore_id, model, None, pipeline_params, note,
                                params_format, delete_model_after_save, stage, mode)
            return

        stage, mode = self._resolve_stage_mode(fedcore_id, model_id, stage, mode)

        self._save_checkpoint_and_record(fedcore_id, model_id, model, None,
                                         delete_model_after_save, stage, mode)

    def update_metrics(self, fedcore_id: str, model_id: str, metrics: dict,
                       stage: Optional[str] = None, mode: Optional[str] = None, trainer=None):
        self.storage.update_record(fedcore_id, model_id, metrics, stage=stage, mode=mode, trainer=trainer)

    def save_metrics_from_evaluator(self, solver, fedcore_id: str, model_id: str):
        metrics_df = self.metrics_tracker.collect_metrics_from_history(solver=solver)

        if not metrics_df.empty and len(metrics_df) > 0:
            last_gen_metrics = metrics_df.iloc[-1].to_dict()
            last_gen_metrics.pop('generation', None)

            if last_gen_metrics:
                self.update_metrics(fedcore_id, model_id, last_gen_metrics)
                self.logger.info("Saved optimization metrics from evaluator to registry")

    def get_latest_record(self, fedcore_id: str, model_id: str) -> Optional[dict]:
        return self.storage.get_latest_record(fedcore_id, model_id)

    def get_model_history(self, fedcore_id: str, model_id: str):
        return self.storage.get_records(fedcore_id, model_id)

    def get_best_checkpoint(self, fedcore_id: str, metric_name: str, mode: str = "max") -> Optional[dict]:
        df = self.storage.load(fedcore_id)
        return self.metrics_tracker.find_best_checkpoint(df, metric_name, mode)

    def get_checkpoint_path(self, fedcore_id: str, model_id: str) -> Optional[str]:
        """Get checkpoint path for a registered model.

        Args:
            fedcore_id: FedCore instance identifier
            model_id: Model identifier

        Returns:
            Checkpoint path string or None if not found
        """
        latest = self.storage.get_latest_record(fedcore_id, model_id)
        return latest.get('checkpoint_path') if latest else None

    def load_model_from_latest_checkpoint(self, fedcore_id: str, model_id: str,
                                          device: torch.device = None) -> Optional[torch.nn.Module]:
        latest = self.storage.get_latest_record(fedcore_id, model_id)
        return (self.checkpoint_manager.load_from_file(latest['checkpoint_path'], device)
                if latest and latest.get('checkpoint_path') else None)

    def list_models(self, fedcore_id: str) -> list:
        return self.storage.list_model_ids(fedcore_id)

    def get_model_with_fallback(self, fedcore_id: str, model_id: str,
                                fallback_model=None, device: torch.device = None):
        loaded_model = self.load_model_from_latest_checkpoint(fedcore_id, model_id, device)
        return loaded_model if loaded_model is not None else fallback_model

    def _delete_model_from_memory(self, model) -> None:
        if model is None:
            return

        if hasattr(model, 'cpu'):
            model.cpu()

        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.grad = None

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_memory_stats(self) -> dict:
        return self.checkpoint_manager.get_gpu_memory_stats()

    def force_cleanup(self) -> None:
        self.checkpoint_manager._cleanup_gpu_memory()
        self.logger.info("Forced GPU memory cleanup executed")

    def cleanup_fedcore_instance(self, fedcore_id: str, compressor_object=None) -> None:
        self.logger.info(f"Starting comprehensive cleanup for fedcore_id={fedcore_id}")
        mem_before = self.get_memory_stats()
        self.logger.info(f"Memory before cleanup: {mem_before.get('allocated_gb', 0):.4f} GB")

        if compressor_object is not None:
            compressor_object = self._extract_fitted_operation(compressor_object)
            if compressor_object is not None:
                self._clean_compressor_models(compressor_object)

        self._clean_registry_storage(fedcore_id)
        self._cleanup_gpu_memory()

        mem_after = self.get_memory_stats()
        memory_freed = mem_before.get('allocated_gb', 0) - mem_after.get('allocated_gb', 0)
        if mem_before.get('allocated_gb', 0) > 0:
            efficiency = (memory_freed / mem_before.get('allocated_gb', 0)) * 100
            self.logger.info(f"Cleanup efficiency: {efficiency:.1f}%")

        self.logger.info("Comprehensive cleanup completed")

    def _extract_fitted_operation(self, compressor_object):
        if Pipeline is None or not isinstance(compressor_object, Pipeline):
            return compressor_object

        if hasattr(compressor_object, 'operator') and hasattr(compressor_object.operator, 'root_node'):
            return getattr(compressor_object.operator.root_node, 'fitted_operation', None)

        return None

    def _clean_compressor_models(self, compressor_object):
        for attr in _MODEL_ATTRS_TO_CLEAN:
            model = getattr(compressor_object, attr, None)
            if model is not None:
                self._delete_model_from_memory(model)
                setattr(compressor_object, attr, None)

        if hasattr(compressor_object, 'trainer') and compressor_object.trainer is not None:
            self._clean_trainer(compressor_object.trainer)
            del compressor_object.trainer
            compressor_object.trainer = None
            gc.collect()

        self._clean_dynamic_models(compressor_object)

    def _clean_trainer(self, trainer):
        for attr_name in ['model', '_trainer']:
            trainer_obj = getattr(trainer, attr_name, None)
            if trainer_obj is not None:
                if attr_name == '_trainer' and hasattr(trainer_obj, 'model'):
                    model = getattr(trainer_obj, 'model', None)
                    if model is not None:
                        self._delete_model_from_memory(model)
                        trainer_obj.model = None
                    del trainer._trainer
                    trainer._trainer = None
                elif attr_name == 'model':
                    self._delete_model_from_memory(trainer_obj)
                    trainer.model = None

    def _clean_dynamic_models(self, obj):
        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue

            attr = getattr(obj, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                self._delete_model_from_memory(attr)
                setattr(obj, attr_name, None)
            elif isinstance(attr, (list, tuple)):
                for i, item in enumerate(attr):
                    if isinstance(item, torch.nn.Module):
                        self._delete_model_from_memory(item)
                        attr[i] = None
            elif isinstance(attr, dict):
                for key, value in list(attr.items()):
                    if isinstance(value, torch.nn.Module):
                        self._delete_model_from_memory(value)
                        attr[key] = None

    def _clean_registry_storage(self, fedcore_id: str):
        df = self.storage.load(fedcore_id)
        if not df.empty and 'checkpoint_bytes' in df.columns:
            df['checkpoint_bytes'] = None
            self.storage.save(fedcore_id, df)

    def _cleanup_gpu_memory(self):
        self.checkpoint_manager._cleanup_gpu_memory()
        if torch.cuda.is_available():
            for _ in range(_CLEANUP_ITERATIONS):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        for _ in range(_CLEANUP_ITERATIONS):
            gc.collect()