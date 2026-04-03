"""
LLM Trainer implementation using transformers library
Real integration with transformers.Trainer
"""
import torch
import logging
from typing import Any, Dict, Optional, Iterable, Union
from enum import Enum
from tqdm import tqdm
import logging

# Transformers imports
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from datasets import Dataset
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.api.utils.checker_rules import DataLoaderHandler
from fedot.industrial.core.models.nn.utils._base import BaseTrainer
from fedot.industrial.core.models.nn.utils.hook_runtime_rules import (
    build_hook_runtime_payload,
    resolve_stage_hooks,
    should_stop_training,
)
from fedot.industrial.core.models.nn.utils.hook_registration_rules import (
    build_hook_registration_plan,
    instantiate_hook_plan,
)
from fedot.industrial.core.models.nn.utils.hooks_collection import HooksCollection
from fedot.industrial.core.models.nn.utils.hooks import LoggingHooks, ModelLearningHooks


class FedCoreTransformersTrainer(TrainerCallback):
    """
    Transformers Callback with FedCore hooks integration.

    Combines HuggingFace Transformers callbacks with FedCore's hook system for
    OptimizerGen, SchedulerRenewal, and other training hooks.
    """

    def __init__(
            self,
            model=None,
            hooks_collection: Optional[HooksCollection] = None,
            hooks_params: Optional[Dict] = None,
            additional_hooks: Optional[Iterable[Enum]] = None,
    ):
        super().__init__()
        self.model = model
        self.hooks_collection = hooks_collection or HooksCollection()
        self.hooks_params = hooks_params or {}
        self._hooks = [LoggingHooks, ModelLearningHooks]
        self._additional_hooks = list(additional_hooks or ())

        self.trainer_objects = {
            'model': model,
            'optimizer': None,
            'scheduler': None,
            'stop': False
        }

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        if self.hooks_params:
            self._init_hooks()

    def _init_hooks(self):
        """Initialize FedCore hooks based on parameters"""
        hook_plan = build_hook_registration_plan(self._hooks, self._additional_hooks, self.hooks_params)
        for hook_instance in instantiate_hook_plan(hook_plan, self.hooks_params, self.model):
            self.hooks_collection.append(hook_instance)

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the beginning of each epoch"""
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        trainer = kwargs.get('trainer')

        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler

        start_payload = build_hook_runtime_payload(
            trainer_objects=self.trainer_objects,
            history=self.history,
        )
        for hook in resolve_stage_hooks(self.hooks_collection, 'start'):
            hook(epoch=epoch, **start_payload)

        if trainer:
            if self.trainer_objects.get('optimizer'):
                trainer.optimizer = self.trainer_objects['optimizer']
            if self.trainer_objects.get('scheduler'):
                trainer.lr_scheduler = self.trainer_objects['scheduler']

        if should_stop_training(self.trainer_objects):
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        """Execute FedCore hooks at the end of each epoch"""
        epoch = state.epoch if hasattr(state, 'epoch') else 0
        trainer = kwargs.get('trainer')

        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.history['train_loss'].append((epoch, latest_log['loss']))
            if 'eval_loss' in latest_log:
                self.history['val_loss'].append((epoch, latest_log['eval_loss']))

        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler

        val_loader = None
        criterion = None
        if trainer:
            if hasattr(trainer, 'eval_dataset') and trainer.eval_dataset:
                val_loader = trainer.get_eval_dataloader(trainer.eval_dataset)
            if hasattr(trainer, 'compute_loss'):
                criterion = trainer.compute_loss

        end_payload = build_hook_runtime_payload(
            trainer_objects=self.trainer_objects,
            history=self.history,
            val_loader=val_loader,
            criterion=criterion,
        )
        for hook in resolve_stage_hooks(self.hooks_collection, 'end'):
            hook(epoch=epoch, **end_payload)

        if trainer:
            if self.trainer_objects.get('optimizer'):
                trainer.optimizer = self.trainer_objects['optimizer']
            if self.trainer_objects.get('scheduler'):
                trainer.lr_scheduler = self.trainer_objects['scheduler']

        if should_stop_training(self.trainer_objects):
            control.should_training_stop = True

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Track learning rate and sync trainer objects on each step"""
        trainer = kwargs.get('trainer')
        if trainer:
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                self.trainer_objects['optimizer'] = trainer.optimizer
                if hasattr(trainer.optimizer, 'param_groups'):
                    current_lr = trainer.optimizer.param_groups[0]['lr']
                    current_step = state.global_step if hasattr(state, 'global_step') else 0
                    self.history['learning_rates'].append((current_step, current_lr))

            if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler:
                self.trainer_objects['scheduler'] = trainer.lr_scheduler

        return control

    def on_init_end(self, args, state, control, **kwargs):
        """Called after Trainer initialization - setup optimizer and scheduler via hooks"""
        trainer = kwargs.get('trainer')
        if trainer:
            self.trainer_objects['trainer'] = trainer
            self.trainer_objects['model'] = trainer.model

        return control


class LLMTrainer(BaseTrainer):
    """
    LLM Trainer that implements our interfaces with real transformers.Trainer integration
    """

    DEFAULT_TRAINING_ARGS = {
        'output_dir': './llm_output',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'warmup_steps': 0,
        'lr_scheduler_type': 'linear',
        'weight_decay': 0.01,
        'logging_dir': './logs',
        'logging_steps': 10,
        'save_steps': 1000,
        'eval_steps': 1000,
        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'no_cuda': False,
    }

    ALLOWED_TRAINING_ARGS = {
        'output_dir', 'num_train_epochs', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'warmup_steps', 'lr_scheduler_type',
        'weight_decay', 'logging_dir', 'logging_steps', 'save_steps',
        'eval_steps', 'evaluation_strategy', 'save_strategy',
        'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better',
        'no_cuda'
    }

    def __init__(self, params: Optional[Dict] = None, **kwargs):
        # if model is None and params and isinstance(params.get('custom_learning_params'), dict):
        #     nested = params.get('custom_learning_params')
        #     model = nested.get('model', model)

        super().__init__(params=params)
        self.model = self.params.get("model", None)

        self.default_training_args = self.DEFAULT_TRAINING_ARGS.copy()
        if params:
            training_params = {k: v for k, v in params.items()
                               if k not in ['model', 'tokenizer', 'custom_learning_params']}
            self.default_training_args.update(training_params)

        self._trainer = None
        self._training_args = None
        self._data_collator = None
        self._fedcore_callback = None
        self._hooks_initialized = False
        self.task_type = None

    @property
    def epochs(self) -> int:
        """Get the number of training epochs."""
        if self._training_args is not None:
            return int(self._training_args.num_train_epochs)
        return int(self.default_training_args.get('num_train_epochs', 3))

    def _init_hooks(self) -> None:
        """
        Initialize hooks for the model.

        Marks hooks as ready to be initialized in FedCoreTransformersTrainer.
        Actual hook creation happens when FedCoreTransformersTrainer is instantiated.
        """
        if self._hooks_initialized:
            return

        self._hooks_initialized = True

    @property
    def epochs(self) -> int:
        """Get the number of training epochs."""
        if self._training_args is not None:
            return int(self._training_args.num_train_epochs)
        return int(self.default_training_args.get('num_train_epochs', 3))

    def _prepare_data(self, input_data) -> Dict[str, Dataset]:
        """Convert InputData/CompressionInputData to transformers Dataset format"""
        train_data = self._dataloader_to_dataset(input_data.train_dataloader)
        eval_data = self._dataloader_to_dataset(input_data.val_dataloader)

        return {
            'train_dataset': train_data,
            'eval_dataset': eval_data
        }

    def _dataloader_to_dataset(self, dataloader) -> Dataset:
        """
        Convert DataLoader to HuggingFace Dataset.

        Dataset reinstantiation is needed because:
        - Transformers Trainer requires HuggingFace Dataset, not PyTorch DataLoader
        - DataLoader is stateful (iterator) and incompatible with Transformers' internal data handling
        - Dataset format allows Transformers to manage batching, shuffling, and distributed training
        - Converts from our batch formats (tuples/dicts) to standard Dataset.__getitem__ format
        """
        data = []
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                labels = batch.get('labels') if 'labels' in batch else batch.get('targets')
                attention_mask = batch.get('attention_mask')

                if input_ids is None:
                    continue

                batch_size = input_ids.shape[0]
                for i in range(batch_size):
                    data.append({
                        'input_ids': input_ids[i].cpu().tolist(),
                        'labels': None if labels is None else labels[i].cpu().tolist(),
                        'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                    })
            elif isinstance(batch, (list, tuple)):
                # Expecting (inputs, labels, attention_mask) style tuples
                inputs = batch[0] if len(batch) >= 1 else None
                labels = batch[1] if len(batch) >= 2 else None
                attention_mask = batch[2] if len(batch) >= 3 else None

                if inputs is None:
                    continue

                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    data.append({
                        'input_ids': inputs[i].cpu().tolist(),
                        'labels': None if labels is None else labels[i].cpu().tolist(),
                        'attention_mask': None if attention_mask is None else attention_mask[i].cpu().tolist(),
                    })
        return Dataset.from_list(data)

    def _create_trainer(self, datasets: Dict[str, Dataset]):
        """Create transformers trainer with FedCore integration"""
        if self.model is None and self._trainer is not None:
            self.model = self._trainer.model

        if self.model is None:
            raise ValueError(
                "LLMTrainer initialization failed: model is None after parameter resolution. Ensure 'model' or 'initial_assumption' is provided in config (including inside 'custom_learning_params').")

        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model = self.model.to(device)

        if not self._hooks_initialized:
            self._init_hooks()

        filtered_args = self._normalize_kwargs(self.default_training_args, self.ALLOWED_TRAINING_ARGS)
        self._training_args = TrainingArguments(**filtered_args)

        hooks_params = self.default_training_args.copy()
        if self.params:
            for key in ['optimizer', 'scheduler', 'criterion', 'learning_rate']:
                if key in self.params:
                    hooks_params[key] = self.params[key]

        self._fedcore_callback = FedCoreTransformersTrainer(
            model=self.model,
            hooks_params=hooks_params,
            additional_hooks=self._additional_hooks
        )

        callbacks = [self._fedcore_callback, EarlyStoppingCallback(early_stopping_patience=3)]

        self._trainer = Trainer(
            model=self.model,
            args=self._training_args,
            train_dataset=datasets.get('train_dataset'),
            eval_dataset=datasets.get('eval_dataset'),
            data_collator=self._data_collator,
            callbacks=callbacks
        )

        self.trainer_objects['trainer'] = self._trainer

    def fit(self, input_data,
            supplementary_data: Optional[Dict] = None, loader_type: str = 'train') -> Any:
        """
        Train the model using InputData/CompressionInputData.

        Input can be either InputData (FEDOT) or CompressionInputData (FedCore).
        """
        logging.info(f"Training LLM model with {loader_type} data...")

        # Final fallback: pull model from input_data if still missing
        if self.model is None:
            candidate_model = getattr(input_data, 'target', None)
            if candidate_model is None and hasattr(input_data, 'target'):
                candidate_model = input_data.target
            if candidate_model is not None:
                self.model = candidate_model

        datasets = self._prepare_data(input_data)
        self._create_trainer(datasets)
        # self.execute_hooks('start', epoch=0)

        train_result = self._trainer.train()
        logging.info(f"Training completed. Loss: {getattr(train_result, 'training_loss', None)}")
        if self._fedcore_callback:
            self.history.update(self._fedcore_callback.history)

        self.model = self._trainer.model

        return self.model

    def predict_for_fit(self, input_data,
                        output_mode: str = "default") -> Any:
        """Make predictions during training"""
        return self._predict_model(input_data, output_mode)

    def predict(self, input_data,
                output_mode: str = "default") -> Any:
        """Make predictions using InputData/CompressionInputData"""

        has_val_loader = hasattr(input_data, 'val_dataloader')

        if self._trainer is not None and has_val_loader:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self._trainer.model = self._trainer.model.to(device)
                for buffer in self._trainer.model.buffers():
                    buffer.data = buffer.data.to(device)
            self._trainer.model.eval()

            eval_dataset = self._dataloader_to_dataset(input_data.val_dataloader)
            return self._trainer.predict(eval_dataset)
        elif self._trainer is None and has_val_loader:
            if self.model is None:
                raise ValueError(
                    "Cannot create trainer for prediction: model is None. Call fit() first or provide model in initialization.")
            datasets = self._prepare_data(x_test)
            self._create_trainer(datasets)
            eval_dataset = self._dataloader_to_dataset(input_data.val_dataloader)
            return self._trainer.predict(eval_dataset)

        predictions_output = self._predict_model(input_data, output_mode)
        pred_values = torch.tensor(predictions_output.predictions)

        output_data = CompressionOutputData(
            # idx=torch.arange(len(pred_values)),
            task=input_data.task,
            predict=pred_values,
            target=None,
            data_type=DataTypesEnum.table,
        )

        return output_data

    def predict_for_fit(self, input_data: Union[InputData],
                        output_mode: str = "default") -> Any:
        """Make predictions during training"""
        return self.predict(input_data, output_mode)

    def save_model(self, path: str) -> None:
        """Save the model using transformers approach"""
        logging.info(f"Saving LLM model to {path}...")

        if self._trainer is not None:
            self._trainer.save_model(path)
        else:
            super().save_model(path)

    def load_model(self, path: str) -> None:
        """Load the model using transformers approach"""
        logging.info(f"Loading LLM model from {path}...")

        super().load_model(path)

    @torch.no_grad()
    def _predict_model(
            self, x_test: Union[InputData], output_mode: str = "default"
    ):
        model: torch.nn.Module = self.model or x_test.target
        model.eval()
        prediction = []

        dataloader = DataLoaderHandler.check_convert(
            getattr(x_test, 'test_dataloader', None) or getattr(x_test, 'val_dataloader', None),
            mode=self.batch_type,
            max_batches=self.calib_batch_limit
        )

        if self.task_type is None:
            if hasattr(x_test, 'task'):
                self.task_type = x_test.task.task_type if hasattr(x_test.task, 'task_type') else x_test.task
            elif hasattr(x_test, 'features') and hasattr(x_test.features, 'task'):
                task_obj = x_test.features.task
                self.task_type = task_obj.task_type if hasattr(task_obj, 'task_type') else task_obj

        for i, batch in tqdm(enumerate(dataloader, 1), total=len(dataloader)):
            if isinstance(batch, dict):
                inputs_dict = {}
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                inputs_embeds = batch.get('inputs_embeds')

                if input_ids is not None:
                    inputs_dict['input_ids'] = input_ids.to(self.device)
                    if attention_mask is not None:
                        inputs_dict['attention_mask'] = attention_mask.to(self.device)
                elif inputs_embeds is not None:
                    inputs_dict['inputs_embeds'] = inputs_embeds.to(self.device)
                    if attention_mask is not None:
                        inputs_dict['attention_mask'] = attention_mask.to(self.device)

                pred = model(**inputs_dict) if len(inputs_dict) > 0 else model()
            else:
                seq = list(batch)
                if len(seq) >= 2 and hasattr(seq[-1], 'dtype'):
                    seq_inputs = seq[:-1]
                else:
                    seq_inputs = seq

                inputs_dict = {}
                if len(seq_inputs) >= 1 and hasattr(seq_inputs[0], 'dtype'):
                    t0 = seq_inputs[0].to(self.device)
                    if t0.dtype in (torch.int32, torch.int64):
                        inputs_dict['input_ids'] = t0
                    else:
                        inputs_dict['inputs_embeds'] = t0
                if len(seq_inputs) >= 2 and hasattr(seq_inputs[1], 'dtype'):
                    t1 = seq_inputs[1].to(self.device)
                    if 'inputs_embeds' not in inputs_dict and t1.dtype in (torch.int32, torch.int64, torch.bool):
                        inputs_dict['attention_mask'] = t1

                pred = model(**inputs_dict)

            pred_tensor = getattr(pred, 'logits', pred)
            prediction.append(pred_tensor)

            if i % getattr(self, '_clear_each', 10) == 0:
                self._clear_cache()

        return self._convert_predict(torch.cat(prediction), output_mode, x_test)

    def _convert_predict(self, pred: Union[torch.Tensor, np.ndarray], output_mode: str = "default",
                         input_data: Union[CompressionInputData, InputData] = None) -> CompressionOutputData:
        """Convert predictions to CompressionOutputData format"""
        if isinstance(pred, torch.Tensor):
            pred_values = pred.cpu().detach()
        elif isinstance(pred, np.ndarray):
            pred_values = torch.from_numpy(pred)
        elif isinstance(pred, list):
            pred_values = torch.tensor(pred)
        else:
            try:
                pred_values = torch.tensor(pred)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Prediction conversion failed: cannot convert {type(pred).__name__} to Tensor. Error: {e}")

        extracted_fields = self._extract_output_fields(input_data)

        if self.task_type is None and input_data is not None:
            if hasattr(input_data, 'task'):
                self.task_type = input_data.task.task_type if hasattr(input_data.task, 'task_type') else input_data.task
            elif hasattr(input_data, 'features') and hasattr(input_data.features, 'task'):
                task_obj = input_data.features.task
                self.task_type = task_obj.task_type if hasattr(task_obj, 'task_type') else task_obj

        checkpoint_info = self._register_model_checkpoint(
            model=self.model,
            stage='after'
        )

        predict = CompressionOutputData(
            features=extracted_fields['features'],
            task=self.task_type,
            predict=pred_values,
            num_classes=extracted_fields['num_classes'],
            train_dataloader=extracted_fields['train_dataloader'],
            val_dataloader=extracted_fields['val_dataloader'],
            data_type=DataTypesEnum.table,
            model=self.model,
            checkpoint_path=checkpoint_info['checkpoint_path'],
            model_id=checkpoint_info['model_id'],
            fedcore_id=checkpoint_info['fedcore_id'],
        )

        return predict

    @property
    def scheduler(self) -> Any:
        """Get scheduler from transformers trainer"""
        if self._fedcore_callback and 'scheduler' in self._fedcore_callback.trainer_objects:
            return self._fedcore_callback.trainer_objects['scheduler']
        return None
