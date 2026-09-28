from collections import OrderedDict
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d, ReLU

from fedot.industrial.core.models.nn.network_modules.other import Sequential


class EncoderDecoderFabric:
    def __init__(self, params: Optional[OperationParameters] = None):
        self.layers = params.get('num_layers', 2)
        self.latent_layer_params = params.get('latent_layer', 16)
        self.mode = params.get('mode', 'multichannel')
        self.convolutional_params = params.get('convolutional_params',
                                               dict(kernel_size=3, stride=0, padding=0))
        self.activation_func = params.get('act_func', ReLU)
        self.dropout_rate = params.get('dropout_rate', 0.5)

    def _get_layer(self, stage: str = 'encoder', mode: str = 'channel_independent'):
        if stage.__contains__('encoder'):
            layer_dict = {'channel_independent': Conv1d,
                          'multichannel': Conv2d}
        else:
            layer_dict = {'channel_independent': ConvTranspose1d,
                          'multichannel': ConvTranspose2d}
        return layer_dict[mode]

    def _get_layer_params(self, layer_number):
        if layer_number == 0:
            in_channels = self.n_steps
            out_channels = self.latent_layer_params
        elif layer_number == self.layers - 1:
            out_channels = self.latent_layer_params
        else:
            out_channels = 32
        conv_params = dict(in_channels=in_channels,
                           out_channels=out_channels) | self.convolutional_params
        return conv_params

    def build(self, stage):
        layer_dict = OrderedDict()
        for layer_number in range(self.layers):
            layer = self._get_layer(stage, self.mode)
            layer_params = self._get_layer_params(layer_number)
            layer_dict.update({f'conv{layer_number}': layer(**layer_params)})
            layer_dict.update({f'relu{layer_number}': self.activation_func()})
        return Sequential(layer_dict)
