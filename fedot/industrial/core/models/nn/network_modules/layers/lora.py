import math
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim

        self.in_features = in_features
        self.out_features = out_features

        def update_layer(
                self,
                adapter_name,
                r,
                lora_alpha,
                lora_dropout,
                init_lora_weights,
        ):
            # This code works for linear layers, override for other layer types
            if r <= 0:
                raise ValueError(
                    f"`r` should be a positive integer value but the value passed is {r}")

            self.r[adapter_name] = r
            self.lora_alpha[adapter_name] = lora_alpha
            if lora_dropout > 0.0:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.Identity()

            self.lora_dropout.update(nn.ModuleDict(
                {adapter_name: lora_dropout_layer}))
            # Actual trainable parameters
            self.lora_A[adapter_name] = nn.Linear(
                self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(
                r, self.out_features, bias=False)

            if init_lora_weights == "loftq":
                self.loftq_init(adapter_name)
            elif init_lora_weights:
                self.reset_lora_parameters(adapter_name, init_lora_weights)

            self.set_adapter(self.active_adapters)

        def reset_lora_parameters(self, adapter_name, init_lora_weights):
            if init_lora_weights is False:
                return

            if adapter_name in self.lora_A.keys():
                if init_lora_weights is True:
                    nn.init.kaiming_uniform_(
                        self.lora_A[adapter_name].weight, a=math.sqrt(5))
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(
                        self.lora_A[adapter_name].weight,
                        std=1 / self.r[adapter_name])
                else:
                    raise ValueError(
                        f"Unknown initialization {init_lora_weights=}")
                nn.init.zeros_(self.lora_B[adapter_name].weight)
            if adapter_name in self.lora_embedding_A.keys():
                # initialize a the same way as the default for nn.linear and b
                # to zero
                nn.init.zeros_(self.lora_embedding_A[adapter_name])
                nn.init.normal_(self.lora_embedding_B[adapter_name])

        def _get_weight_norm(
                self,
                weight,
                lora_weight,
                scaling) -> torch.Tensor:
            # calculate L2 norm of weight matrix, column-wise
            weight = weight + scaling * lora_weight
            weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
            return weight_norm

        def _cache_store(self, key: str, value: Any) -> None:
            self._caches[key] = value

        def _cache_pop(self, key: str) -> Any:
            value = self._caches.pop(key)
            return value

        def set_scale(self, adapter, scale):
            if adapter not in self.scaling:
                # Ignore the case where the adapter is not in the layer
                return
            self.scaling[adapter] = scale * \
                self.lora_alpha[adapter] / self.r[adapter]

        def scale_layer(self, scale: float) -> None:
            if scale == 1:
                return

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                self.scaling[active_adapter] *= scale

        def unscale_layer(self, scale=None) -> None:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue

                if scale is None:
                    self.scaling[active_adapter] = self.lora_alpha[active_adapter] / \
                        self.r[active_adapter]
                else:
                    self.scaling[active_adapter] /= scale

        def _check_forward_args(self, x, *args, **kwargs):
            """Check if the arguments are compatible with the configs and state of the model"""
            adapter_names = kwargs.get("adapter_names", None)
            if adapter_names is None:
                return

            if len(x) != len(adapter_names):
                msg = (
                    "Length of `adapter_names` should be the same as the number of inputs, but got "
                    f"{len(adapter_names)} and {len(x)} respectively.")
                raise ValueError(msg)

            if self.merged:
                # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
                # adapters. Therefore, it is better to raise an error in this
                # case.
                msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
                raise ValueError(msg)

            unique_adapters = set(self.active_adapters)
            for adapter_name in unique_adapters:
                if self.use_dora.get(adapter_name, False):
                    msg = "Cannot pass `adapter_names` when DoRA is enabled."
                    raise ValueError(msg)

        def _mixed_batch_forward(
                self,
                x: torch.Tensor,
                *args: Any,
                adapter_names: list[str],
                **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same
            # batch at inference time.
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append(
                    [index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]].to(
                    lora_A.weight.dtype)
                lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
                result[sub_batch_indices_list[i]
                       ] += lora_output.to(torch_result_dtype)

            return result


class Linear(nn.Module):
    # Lora implemented in a dense layer
    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like
            # (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False,
              adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it
                        # to 1 here
                        weight_norm = self._get_weight_norm(
                            orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(
                            f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1,
                                                        1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)

                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)

                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not
        # supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(
            weight_B @ weight_A,
            self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(
            self,
            x: torch.Tensor,
            *args: Any,
            **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoRALayer):
    # LoRA implemented in an Embedding layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(
            self,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights):
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)

        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False,
              adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + \
                        self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + \
                        self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not
        # supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(
            weight_B @ weight_A,
            True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
            self,
            x: torch.Tensor,
            *args: Any,
            adapter_names: list[str],
            **kwargs: Any) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same
        # batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append(
                [index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]
                   ] += (after_A @ embedding_B) * scaling

        return result

    def _embed(
            self,
            input: torch.Tensor,
            weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(
            self,
            x: torch.Tensor,
            *args: Any,
            **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoRALayer):
    # Lora implemented in a conv2d layer
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        LoRALayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
        )

    def update_layer(
            self,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights):
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(
            self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(
            r, self.out_features, (1, 1), (1, 1), bias=False)

        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False,
              adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it
                        # to 1 here
                        weight_norm = self._get_weight_norm(
                            orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(
                            f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1,
                                                        1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)

                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)

                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not
        # supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(
                2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, channel-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight
        # tensors of Conv2D
        weight_norm = weight.norm(
            p=2, dim=(
                1, 2, 3), keepdim=True).transpose(
            1, 0)
        return weight_norm

    def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied.
        This should be added on top of the base layer output.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        lora_weight = torch.mm(
            lora_B.weight.flatten(
                start_dim=1), lora_A.weight.flatten(
                start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.lora_magnitude_vector[active_adapter]
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * (
            F.conv2d(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


class LoRAParametrization(nn.Module):
    """Class responsible for LoRA Update matrices.

    References:
        @inproceedings{
            hu2022lora,
            title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
            author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
            booktitle={International Conference on Learning Representations},
            year={2022},
            url={https://openreview.net/forum?id=nZeVKeeFYf9}
        }
        Original paper: https://arxiv.org/pdf/2106.09685
    """

    def __init__(
            self,
            features_in,
            features_out,
            rank=1,
            alpha=1,
            device="cpu"):
        super().__init__()
        # Section 4.1 from paper:
        #    We use a random Gaussian initialization for A and zero for B,
        #    so ∆W = B * A is zero at the beginning of training
        self.lora_A = nn.Parameter(
            torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))

        nn.init.normal_(self.lora_A, mean=0, std=1)
        # Section 4.1 from paper:
        #   We then scale ∆W * x by α/r , where α is a constant in r. When optimizing with Adam,
        #   tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
        #   As a result, we simply set α to the first r we try and do not tune it.
        # This scaling helps to reduce the need to re-tune hyperparameters when
        # we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B * A) * scale if lora is enabled instead of W * x +
            # ∆W * x
            return original_weights + \
                torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale

        return original_weights


def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights
    #   for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks)
    #   both for simplicity and parameter-efficiency.
    #   [...]
    # We leave the empirical investigation of [...], and biases to a future
    # work.

    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in,
        features_out,
        rank=rank,
        alpha=lora_alpha,
        device=device)
