from typing import Optional
import torch
from torch.nn import Conv2d, Parameter
from torch.nn.functional import conv2d

from fedot.industrial.core.architecture.abstraction.сheckers import parameter_value_check


class DecomposedConv2d(Conv2d):
    """Extends the Conv2d layer by implementing the singular value decomposition of
    the weight matrix.

    Args:
        base_conv:  The convolutional layer whose parameters will be copied
        decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
            If ``None`` create layers without decomposition.
        forward_mode: ``'one_layer'``, ``'two_layers'`` or ``'three_layers'`` forward pass calculation method.
    """

    def __init__(
        self,
        base_conv: Conv2d,
        decomposing_mode: Optional[str] = 'channel',
        forward_mode: str = 'one_layer',
        device=None,
        dtype=None,
    ) -> None:

        parameter_value_check('forward_mode', forward_mode, {
                              'one_layer', 'two_layers', 'three_layers'})

        if forward_mode != 'one_layer':
            assert base_conv.padding_mode == 'zeros', \
                "only 'zeros' padding mode is supported for '{forward_mode}' forward mode."
            assert base_conv.groups == 1, f"only 1 group is supported for '{forward_mode}' forward mode."

        super().__init__(
            base_conv.in_channels,
            base_conv.out_channels,
            base_conv.kernel_size,
            base_conv.stride,
            base_conv.padding,
            base_conv.dilation,
            base_conv.groups,
            (base_conv.bias is not None),
            base_conv.padding_mode,
            device,
            dtype,
        )
        self.load_state_dict(base_conv.state_dict())
        self.forward_mode = forward_mode
        if decomposing_mode is not None:
            self.decompose(decomposing_mode)
        else:
            self.U = None
            self.S = None
            self.Vh = None
            self.decomposing = None

    def __set_decomposing_params(self, decomposing_mode):
        n, c, w, h = self.weight.size()
        decomposing_modes = {
            'channel': {
                'type': 'channel',
                'permute': (0, 1, 2, 3),
                'decompose_shape': (n, c * w * h),
                'compose_shape': (n, c, w, h),
                'U shape': (n, 1, 1, -1),
                'U': {
                    'stride': 1,
                    'padding': 0,
                    'dilation': 1,
                },
                'Vh shape': (-1, c, w, h),
                'Vh': {
                    'stride': self.stride,
                    'padding': self.padding,
                    'dilation': self.dilation,
                }
            },
            'spatial': {
                'type': 'spatial',
                'permute': (0, 2, 1, 3),
                'decompose_shape': (n * w, c * h),
                'compose_shape': (n, w, c, h),
                'U shape': (n, w, 1, -1),
                'U': {
                    'stride': (self.stride[0], 1),
                    'padding': (self.padding[0], 0),
                    'dilation': (self.dilation[0], 1),
                },
                'Vh shape': (-1, c, 1, h),
                'Vh': {
                    'stride': (1, self.stride[1]),
                    'padding': (0, self.padding[1]),
                    'dilation': (1, self.dilation[1]),
                }
            },
        }
        parameter_value_check('decomposing_mode',
                              decomposing_mode, set(decomposing_modes.keys()))
        self.decomposing = decomposing_modes[decomposing_mode]

    def decompose(self, decomposing_mode: str) -> None:
        """Decomposes the weight matrix in singular value decomposition.
        Replaces the weights with U, S, Vh matrices such that weights = U * S * Vh.
        Args:
            decomposing_mode: ``'channel'`` or ``'spatial'`` weights reshaping method.
        Raises:
            ValueError: If ``decomposing_mode`` not in valid values.
        """
        self.__set_decomposing_params(decomposing_mode=decomposing_mode)
        W = self.weight.permute(self.decomposing['permute']).reshape(
            self.decomposing['decompose_shape'])
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.U = Parameter(U)
        self.S = Parameter(S)
        self.Vh = Parameter(Vh)
        self.register_parameter('weight', None)

    def compose(self) -> None:
        """Compose the weight matrix from singular value decomposition.
        Replaces U, S, Vh matrices with weights such that weights = U * S * Vh.
        """
        W = self.U @ torch.diag(self.S) @ self.Vh
        self.weight = Parameter(
            W.reshape(
                self.decomposing['compose_shape']).permute(
                self.decomposing['permute']))
        self.register_parameter('U', None)
        self.register_parameter('S', None)
        self.register_parameter('Vh', None)
        self.decomposing = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.decomposing is not None:
            if self.forward_mode == 'one_layer':
                return self._one_layer_forward(input)
            if self.forward_mode == 'two_layers':
                return self._two_layers_forward(input)
            if self.forward_mode == 'three_layers':
                return self._three_layers_forward(input)
        else:
            return self._conv_forward(input, self.weight, self.bias)

    def _one_layer_forward(self, input: torch.Tensor) -> torch.Tensor:
        W = self.U @ torch.diag(self.S) @ self.Vh
        W = W.reshape(self.decomposing['compose_shape']).permute(
            self.decomposing['permute'])
        return self._conv_forward(input, W, self.bias)

    def _two_layers_forward(self, input: torch.Tensor) -> torch.Tensor:
        SVh = (torch.diag(self.S) @ self.Vh).view(self.decomposing['Vh shape'])
        U = self.U.reshape(self.decomposing['U shape']).permute(0, 3, 1, 2)
        x = conv2d(input=input, weight=SVh, groups=self.groups,
                   **self.decomposing['Vh'])
        return conv2d(
            input=x,
            weight=U,
            bias=self.bias,
            **self.decomposing['U'])

    def _three_layers_forward(self, input: torch.Tensor) -> torch.Tensor:
        S = torch.diag(self.S).view([len(self.S), len(self.S), 1, 1])
        Vh = self.Vh.view(self.decomposing['Vh shape'])
        U = self.U.view(self.decomposing['U shape']).permute(0, 3, 1, 2)
        x = conv2d(input=input, weight=Vh, groups=self.groups,
                   **self.decomposing['Vh'])
        x = conv2d(input=x, weight=S, padding=0)
        return conv2d(
            input=x,
            weight=U,
            bias=self.bias,
            **self.decomposing['U'])

    def set_U_S_Vh(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
            vh: torch.Tensor) -> None:
        """Update U, S, Vh matrices.
        Raises:
            Assertion Error: If ``self.decomposing`` is False.
        """
        assert self.decomposing is not None, "for setting U, S and Vh, the model must be decomposed"
        self.U = Parameter(u)
        self.S = Parameter(s)
        self.Vh = Parameter(vh)
