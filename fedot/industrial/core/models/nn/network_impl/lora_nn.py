from typing import Optional

import torch
import torch.nn.utils.parametrize as parametrize
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor
from torch import nn, optim

from fedot.industrial.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_impl.common_model.dummy_nn import DummyOverComplicatedNeuralNetwork
from fedot.industrial.core.models.nn.network_impl.common_model.inception import InceptionTimeModel
from fedot.industrial.core.models.nn.network_impl.common_model.resnet import ResNetModel
from fedot.industrial.core.models.nn.network_impl.feature_extraction.explainable_convolution_model import XCModel
from fedot.industrial.core.models.nn.network_impl.forecasting_model.nbeats import NBeatsModel
from fedot.industrial.core.models.nn.network_impl.forecasting_model.tst import TSTModel
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.power_iteration_decomposition import \
    RSVDDecomposition

NEURAL_MODEL = {
    # fundamental models
    'inception_model': InceptionTimeModel,
    'resnet_model': ResNetModel,
    'nbeats_model': NBeatsModel,
    # transformer models
    'tst_model': TSTModel,
    # explainable models
    'xcm_model': XCModel,
    # linear_dummy_model
    'dummy': DummyOverComplicatedNeuralNetwork,

}


def linear_layer_parameterization_with_info(updated_weight, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks
    #   and freeze the MLP modules (so they are not trained in downstream tasks)
    #   both for simplicity and parameter-efficiency.
    #   [...]
    # We leave the empirical investigation of [...], and biases to a future
    # work.

    return LoRAParametrizationWithInfo(updated_weight, rank=rank, alpha=lora_alpha)


class LoRAParametrizationWithInfo(nn.Module):
    def __init__(self,
                 updated_weight,
                 rank=1,
                 alpha=1):
        super().__init__()

        self.rank = rank
        self.lora_B, self.lora_A = updated_weight[0], updated_weight[1]
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            return (
                original_weights +
                torch.matmul(
                    self.lora_B,
                    self.lora_A).view(
                    original_weights.shape) *
                self.scale).to(
                torch.float32)
        else:
            return original_weights


class LoraModel(BaseNeuralModel):
    """Class responsible for Low Rank adaptation model implementation.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        # hyperparams for LoRa learning
        self.epochs = self.params.get('epochs', 1)
        self.batch_size = self.params.get('epochs', 16)
        self.model_type = self.params.get('neural_architecture', 'dummy')
        self.pretrain = self.params.get('from_pretrain', False)
        self.lora_init = self.params.get('lora_init', 'random')
        # hyperparams for SVD
        self.sampling_share = self.params.get('sampling_share', 0.3)
        self.rank = self.params.get('lora_rank', 1)
        self.power_iter = self.params.get('power_iter', 3)
        self.use_approx = self.params.get('use_rsvd', True)

        svd_params = dict(rank=self.rank,
                          sampling_share=self.sampling_share,
                          power_iter=self.power_iter)
        self.industrial_impl = NEURAL_MODEL[self.model_type]
        self.rsvd = RSVDDecomposition(OperationParameters(**svd_params))

    def __repr__(self):
        return f'LoRa - {self.model_type}'

    def _save_and_clear_cache(self):
        pass

    def _init_model(self, input_data):
        self.model = self.industrial_impl(
            input_dim=input_data.features.shape[1],
            output_dim=self.num_classes).to(
            default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = self._get_loss_metric(input_data)
        return loss_fn, optimizer

    def __init_lora(self, spectrum_by_layer):
        updated_weight = []
        if self.lora_init == 'residual_lora':
            for spectrum in spectrum_by_layer:
                _a = torch.tensor(
                    spectrum[0][:, :self.rank] @ spectrum[1][:self.rank])
                _b = torch.tensor(
                    spectrum[1][:self.rank] @ spectrum[2][:self.rank, :])

                lora_A = nn.Parameter(
                    torch.unsqueeze(
                        _a, 1)).to(
                    default_device())
                lora_B = nn.Parameter(
                    torch.unsqueeze(
                        _b, 0)).to(
                    default_device())
                updated_weight.append((lora_B, lora_A))
        elif self.lora_init == 'random':
            for spectrum in spectrum_by_layer:
                features_in = spectrum[0].shape[0]
                features_out = spectrum[2].shape[0]
                lora_A = nn.Parameter(
                    torch.zeros(
                        (self.rank, features_out)).to(
                        default_device()))
                lora_B = nn.Parameter(
                    torch.zeros(
                        (features_in, self.rank)).to(
                        default_device()))
                lora_B = nn.init.normal_(lora_B, mean=0, std=1)
                updated_weight.append((lora_B, lora_A))
        elif self.lora_init == 'residual_core':
            for spectrum in spectrum_by_layer:
                _a = torch.tensor(
                    spectrum[0][:, self.rank:] @ spectrum[1][self.rank:])
                _b = torch.tensor(
                    spectrum[1][self.rank:] @ spectrum[2][self.rank:, :])
                lora_A = nn.Parameter(
                    torch.unsqueeze(
                        _a, 1)).to(
                    default_device())
                lora_B = nn.Parameter(
                    torch.unsqueeze(
                        _b, 0)).to(
                    default_device())
                updated_weight.append((lora_B, lora_A))
        return updated_weight

    def _evaluate_decomposition(self):
        spectrum_by_layer = []
        for name, param in self.model.named_parameters():
            if name.__contains__('weight'):
                spectrum_by_layer.append(
                    self.rsvd.rsvd(
                        tensor=param.clone().cpu().detach().numpy(),
                        approximation=self.use_approx))

        return spectrum_by_layer

    def _create_lora(self, updated_weight):

        parametrize_by_layer = [
            parametrize.register_parametrization(
                self.model.linear1,
                "weight",
                linear_layer_parameterization_with_info(
                    weight,
                    default_device(),
                    self.rank),
                unsafe=True) for weight in updated_weight]

    def enable_disable_lora(self, enabled=True):
        for name, param in self.model.named_parameters():
            param.parametrizations["weight"][0].enabled = enabled
            if "lora" not in name:
                print(f"Freezing non-LoRA parameter {name}")
                param.requires_grad = False

    def _fit_model(self, input_data: InputData, split_data: bool = False):
        loss_fn, optimizer = self._init_model(input_data)
        train_loader, val_loader = self._prepare_data(input_data, split_data)

        if not self.pretrain:
            self._train_loop(
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer
            )
        spectrum_by_layer = self._evaluate_decomposition()
        updated_weight = self.__init_lora(spectrum_by_layer)
        self._create_lora(updated_weight)

    @convert_to_3d_torch_array
    def _predict_model(self, x_test, output_mode: str = 'default'):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device('cpu'))
        pred = self.model(x_test)
        return self._convert_predict(pred, output_mode)
