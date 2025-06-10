from typing import Callable

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    GroupNorm,
    Dropout
)

from neural_nets.activations import get_activation

from utils.initializers import _get_initializer, ones_


class SeparableInvertResidual(Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        expand_factor: int,
        stride: int = 1,
        num_groups_norm: int = 4,
        drop_p: float = 0.3,
        activation: str = "swish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        *args,
        **kwargs
    ):
        r"""
        Implementation of Inverted Residual block, which is introduced in MobileNetv2.

        Args:
            in_channels (int): Number of channels of input.
            out_channels (int): Number of channels of output.
            expand_factor (int, optional): Expand factor of the first pointwise conv.
                Defaults to 3.
            num_groups_norm (int, optional): Number of group for GN layer. Defaults to 4.
            activation (str, optional): Type of nonlinear activation function.
                Defaults to "swish".
        """
        super().__init__() 
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        expand_channels = expand_factor*in_channels
        if in_channels == out_channels and stride == 1:
            self.is_residual = True
        else:
            self.is_residual = False

        # pointwise
        expansion = [
            Dropout(drop_p),
            Conv2d(
               in_channels=in_channels, 
               out_channels=expand_channels,
               kernel_size=1,
               **factory_kwargs
            ),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=expand_channels,
                **factory_kwargs
            ),
            get_activation(activation, **factory_kwargs),
        ]
        self.expansion = Sequential(*expansion)
        
        # depthwise
        depthwise = [
            Dropout(drop_p),
            Conv2d(
                in_channels=expand_channels, 
                out_channels=expand_channels,
                kernel_size=3,
                stride=stride,
                padding=(1, 1),
                groups=expand_channels,
                **factory_kwargs
            ),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=expand_channels,
                **factory_kwargs
            ),
            get_activation(activation, **factory_kwargs),
        ]
        self.depthwise = Sequential(*depthwise)
        
        # pointwise
        projection = [
            Dropout(drop_p),
            Conv2d(
                in_channels=expand_channels, 
                out_channels=out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=out_channels,
                **factory_kwargs
            ),
        ]
        self.projection = Sequential(*projection)
        
        self._reset_parameters()
    
    
    def _reset_parameters(self) -> None:
        self.initializer(self.expansion[1].weight)
        self.initializer(self.depthwise[1].weight)
        self.initializer(self.projection[1].weight)
        
        if self.expansion[1].bias is not None:
            ones_(self.expansion[1].bias)
            ones_(self.depthwise[1].bias)
            ones_(self.projection[1].bias)
        
        
    def forward(self, input):
        Z = self.projection(self.depthwise(self.expansion(input)))
        if self.is_residual:
            Z = Z + input
        return Z