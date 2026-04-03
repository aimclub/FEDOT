from enum import Enum
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable
from pymonad.maybe import Maybe
import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.tensordata import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import Tensor
from tqdm import tqdm
import logging

from fedot.industrial.api.utils.checker_rules import DataLoaderHandler
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.repository.constanst_repository import (ModelLearningHooks, LoggingHooks,TorchLossesConstant)

from fedot.industrial.core.models.nn.utils.hooks import BaseHook
from fedot.industrial.core.models.nn.utils.hook_runtime_rules import (
    build_hook_runtime_payload,
    resolve_stage_hooks,
    should_stop_training,
)
from fedot.industrial.core.models.nn.utils.hook_registration_rules import (
    build_initialized_hooks,
    resolve_hook_groups,
)
from fedot.industrial.core.models.nn.utils.hooks_collection import HooksCollection
from fedot.industrial.core.models.nn.utils._base import BaseTrainer

BASE_REGRESSION_DTYPE = torch.float32


class BaseNeuralModel(torch.nn.Module, BaseTrainer):
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None, additional_hooks=None):
        torch.nn.Module.__init__(self)
        BaseTrainer.__init__(self, params=params.to_dict() if hasattr(params, 'to_dict') else params)

        self.learning_params = self.params.get('custom_learning_params', {})
        self._init_null_object()
        self._init_empty_object()

        self.epochs = self.params.get("epochs", 1)
        self.batch_size = self.params.get("batch_size", 16)
        self.learning_rate = self.params.get("learning_rate", 0.001)
        self.model = self.params.get("model", None)
        self._init_custom_criterions(
            self.params.get("custom_criterions", {}))  # let it be dict[name : coef], let nodes add it to trainer
        self.criterion = self.__get_criterion()
        self.device = self.params.get('device', default_device())
        self.model_params = self.params.get('model_params', {})
        self._hooks = [LoggingHooks, ModelLearningHooks]
        self.register_additional_hooks(additional_hooks or [])
        self._clear_each = self.learning_params.get('clear_each', 10)

    def _init_custom_criterions(self, custom_criterions: dict):
        for name, coef in custom_criterions.items():
            if hasattr(TorchLossesConstant, name):
                criterion = TorchLossesConstant[name].value
            else:
                raise ValueError(f'Unknown type `{name}` of custom loss')
            self.history[f'train_{name}_loss'] = []
            self.history[f'val_{name}_loss'] = []
            custom_criterions[name] = (criterion(), coef)
        self.custom_criterions = custom_criterions

    def _init_null_object(self):
        self.label_encoder = None
        self.is_regression_task = False
        self.target = None
        self.task_type = None
        self.checkpoint_folder = self.params.get('checkpoint_folder', None)
        self.batch_limit = self.learning_params.get('batch_limit', None)
        self.calib_batch_limit = self.learning_params.get('calib_batch_limit', None)
        self.batch_type = self.learning_params.get('batch_type', None)

    def _init_empty_object(self):
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # keep one source of truth for hook lifecycle state
        self.hooks = self.hooks_collection = HooksCollection()

    def __repr__(self):
        return self.__class__.__name__ + '\n' + repr(self.hooks)

    def _init_model(self):
        pass

    def _init_hooks(self):
        hook_groups = resolve_hook_groups(self._hooks, self._additional_hooks)
        for hook in build_initialized_hooks(hook_groups, self.params, self.model):
            self.hooks.append(hook)

    def __get_criterion(self):
        key = self.params.get('loss', None) or self.params.get('criterion', None)
        if isinstance(key, str):
            return TorchLossesConstant[key].value()
        elif isinstance(key, Callable):
            return key
        else:
            return None

    def __substitute_device_quant(self):
        if not getattr(self.model, '_is_quantized', False):
            self.device = default_device('cpu')
            self.model.to(self.device)
            logging.info('Quantized model inference supports CPU only')

    def fit(self, input_data, supplementary_data: dict = None, loader_type='train'):
        # define data for fit process
        self.custom_fit_process = supplementary_data is not None
        train_loader = input_data.train_dataloader
        val_loader = input_data.val_dataloader
        self.task_type = input_data.task.task_type
        # define model for fit process
        self.model = input_data.target if self.model is None else self.model
        self.optimised_model = self.model
        self.model.to(self.device)
        self._init_hooks()
        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=self.criterion,
        )
        self._clear_cache()
        return self.model

    def _run_one_epoch(self, epoch, dataloader, loss_fn, optimizer):
        training_loss = 0.0
        self.model.train()
        for batch in tqdm(dataloader, desc='Batch #'):
            *inputs, targets = batch
            inputs = tuple(inputs_.to(self.device) for inputs_ in inputs if hasattr(inputs_, 'to'))
            targets = targets.to(self.device)
            output = self.model(*inputs)
            if self.task_type.name == 'regression' or self.task_type.name.__contains__('forecasting'):
                targets = targets.to(BASE_REGRESSION_DTYPE)
                output = output.to(BASE_REGRESSION_DTYPE)
            loss = self._compute_loss(loss_fn, output,
                                      targets.to(self.device), epoch=epoch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        avg_loss = training_loss / len(dataloader)
        self.history['train_loss'].append((epoch, avg_loss))  # changed to match epoch and loss

    def _train_loop(self, train_loader, val_loader, loss_fn):
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        for epoch in range(1, self.epochs + 1):
            start_payload = build_hook_runtime_payload(
                trainer_objects=self.trainer_objects,
                history=self.history,
                learning_rate=self.learning_rate,
            )
            for hook in resolve_stage_hooks(self.hooks, 'start'):
                hook(epoch=epoch, **start_payload)
            if should_stop_training(self.trainer_objects):
                break
            self._run_one_epoch(epoch=epoch,
                                dataloader=train_loader,
                                loss_fn=loss_fn,
                                optimizer=self.optimizer)
            end_payload = build_hook_runtime_payload(
                trainer_objects=self.trainer_objects,
                history=self.history,
                learning_rate=self.learning_rate,
                val_loader=val_loader,
                criterion=partial(self._compute_loss, criterion=loss_fn),
            )
            for hook in resolve_stage_hooks(self.hooks, 'end'):
                hook(epoch=epoch, **end_payload)
            if should_stop_training(self.trainer_objects):
                break
        return self

    def predict(self, input_data, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data, output_mode)

    def predict_for_fit(self, input_data, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data, output_mode)

    @torch.no_grad()
    def _predict_model(
            self, x_test, output_mode: str = "default"
    ):
        model: torch.nn.Module = self.model or x_test.model
        model.eval()
        prediction = []
        dataloader = DataLoaderHandler.check_convert(x_test.val_dataloader,
                                                     mode=self.batch_type,
                                                     max_batches=self.calib_batch_limit)
        if self.task_type is None:
            self.task_type = x_test.task.task_type
        for i, batch in tqdm(enumerate(dataloader, 1), total=len(dataloader)):  ###TODO why val_dataloader???
            *inputs, targets = batch
            inputs = tuple(inputs_.to(self.device) for inputs_ in inputs if hasattr(inputs_, 'to'))
            prediction.append(model(*inputs))
            del inputs
            del batch
            if i % self._clear_each == 0:
                self._clear_cache()
        return self._convert_predict(torch.concat(prediction), output_mode, x_test)

    def _convert_predict(self, pred: Tensor, output_mode: str = "labels",
                         input_data = None):
        assert isinstance(pred, torch.Tensor), "Prediction convertion failed, prediction is not a Tensor"
        have_encoder = all([self.label_encoder is not None, output_mode == "labels"])
        if self.task_type.name == 'regression':
            self.is_regression_task = True
        output_is_clf_labels = all(
            [not self.is_regression_task, output_mode == "labels"]
        )
        pred = Maybe.insert(pred). \
            then(
            lambda predict: predict.cpu().detach().numpy() if self.is_regression_task else F.softmax(predict, dim=1)). \
            then(
            lambda predict: torch.argmax(predict, dim=1).cpu().detach().numpy() if output_is_clf_labels else predict). \
            then(lambda predict: self.label_encoder.inverse_transform(predict) if have_encoder else predict). \
            maybe(None, lambda output: output)

        extracted_fields = self._extract_output_fields(input_data)

        checkpoint_info = self._register_model_checkpoint(
            model=self.model,
            stage='after'
        )

        predict = TensorData(
            features=extracted_fields['features'],
            task=self.task_type,
            predict=pred,
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

    # def _clear_cache(self):
    #     """Clear CUDA cache - shared by BaseNeuralModel and LLMTrainer"""
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()

    # def __wrap(self, model):
    #     if not isinstance(model, BaseNeuralModel):
    #         return BaseNeuralModel.wrap(model, self.params)
    #     return model

    @staticmethod
    def get_validation_frequency(epoch, lr):
        if epoch < 10:
            return 1  # Validate frequently in early epochs
        elif lr < 0.01:
            return 5  # Validate less frequently after learning rate decay
        else:
            return 2  # Default validation frequency

    @property
    def is_quantised(self):
        return getattr(self, '_is_quantised', False)

    @property
    def optimizer(self):
        return self.trainer_objects['optimizer']

    @property
    def scheduler(self):
        return self.trainer_objects['scheduler']


class BaseNeuralForecaster(BaseNeuralModel):
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.train_horizon = self.params.get('train_horizon', 1)
        self.test_horizon = self.params.get('test_horizon', 1)
        self.in_sample_regime = self.params.get('use_in_sample', True)
        self.use_exog_features = self.params.get('use_exog_features', False)
        self.forecasting_blocks = int(self.test_horizon / self.train_horizon)
        self.loss = self.params.get('loss', 'smape')
        self.val_interval = 5

    def out_of_sample_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, target: Tensor):
        pred = self.model(x=tensor_endogen, mask=None)  # output [bs x seq_len x horizon]
        predict = pred[:, -1, :][:, None, :]  # take predict from last point as output [bs x 1 x train_horizon]
        return predict

    def create_features_from_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, predict: Tensor):
        features_from_predict = torch.concat([predict, tensor_exogen], dim=1)
        tensor_endogen = tensor_endogen[:, :, self.train_horizon:]
        tensor_endogen = torch.concat([tensor_endogen, features_from_predict], dim=2)
        return tensor_endogen

    def in_sample_predict(self, tensor_endogen: Tensor, tensor_exogen: Tensor, target: Tensor):
        all_predict = []
        new_tensor_endogen = tensor_endogen
        for block in range(self.forecasting_blocks):
            start_idx, end_idx = block * self.train_horizon, block * self.train_horizon + self.train_horizon
            exog_feature = tensor_exogen[:, :, start_idx:end_idx]
            horizon_pred = self.out_of_sample_predict(new_tensor_endogen, exog_feature,
                                                      target)  # output [bs x 1 x train_horizon]
            all_predict.append(horizon_pred)
            new_tensor_endogen = self.create_features_from_predict(new_tensor_endogen, exog_feature, horizon_pred)
        in_sample_predict = torch.concat(all_predict, dim=2)  # output [bs x 1 x test_horizon]
        return in_sample_predict

    def _run_one_epoch(self, epoch, dataloader, loss_fn, optimizer):
        training_loss = 0.0
        self.model.train()
        for batch in dataloader:
            x_hist, x_fut, y = [b.to(self.device) for b in batch]
            predict = self.out_of_sample_predict(x_hist, x_fut, y)
            loss = loss_fn(predict, y)  # output [bs x last_hist_val x train_horizon]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()
        return optimizer, loss_fn, training_loss

    def _predict_model(self, input_data, output_mode: str = 'default'):
        self.model.eval()

        def predict_loop(batch):
            x_hist, x_fut, y = [b.to(self.device) for b in batch]
            if self.in_sample_regime:
                predict = self.in_sample_predict(x_hist, x_fut, y)
            else:
                predict = self.out_of_sample_predict(x_hist, x_fut, y)
            predict = predict.cpu().detach().numpy().squeeze()
            target = y.cpu().detach().numpy().squeeze()
            return predict, target

        prediction = list(map(lambda batch: predict_loop(batch), input_data.test_dataloader))
        all_prediction = np.concatenate([x[0] for x in prediction])
        # all_target = np.concatenate([x[1] for x in prediction])
        return all_prediction
