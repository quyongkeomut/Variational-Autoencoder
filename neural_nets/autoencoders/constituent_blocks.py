from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    GroupNorm,
    BatchNorm2d,
    ConvTranspose2d,
)

from neural_nets.conv_block import SeparableInvertResidual
from neural_nets.activations import get_activation

from utils.initializers import _get_initializer, ones_


class DownBlock(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        expand_factor: int = 3,
        drop_p: float = 0.3,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        *args,
        **kwargs
    ):  
        """
        Downsample block of lightweight Encoder

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            expand_factor (int, optional): Expand factor used in expansion conv layer
                of inverted residual block . Defaults to 3.
            drop_p (float, optional): Dropout rate. Defaults to 0.3.
            activation (str, optional): Activation function. Defaults to "hardswish".
        """
        super().__init__()
        
        invert_residual_kwargs = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "expand_factor": expand_factor, 
            "drop_p": drop_p,
            "activation": activation,
            "initializer": initializer
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        layers = [                         
            # conv for downsampling 
            Conv2d(
                in_channels, 
                out_channels,
                kernel_size=3,
                stride=2,
                padding=(1, 1),
                **factory_kwargs
            ),
            get_activation(activation),
            Conv2d(
                out_channels, 
                out_channels,
                kernel_size=3,
                padding=(1, 1),
                **factory_kwargs
            ),
            BatchNorm2d(out_channels, **factory_kwargs),
            get_activation(activation),
            SeparableInvertResidual(**invert_residual_kwargs, **factory_kwargs),
        ]
        self.layers = Sequential(*layers)
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)
        self.initializer(self.layers[2].weight)
        if self.layers[0].bias is not None:
            ones_(self.layers[0].bias)
            ones_(self.layers[2].bias)

    def forward(
        self, 
        input: Tensor
    ) -> Tensor:
        """
        Forward method of Down block

        Args:
            input (Tensor): Input representation
                        
        Returns:
            Tensor: Output
        """
        Z = input
        return self.layers(Z)    
    

class UpBlock(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        expand_factor: int = 3,
        drop_p: float = 0.3,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
    ):  
        """
        Upsamole block of lightweight Decoder

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            expand_factor (int, optional): Expand factor used in expansion conv layer
                of inverted residual block . Defaults to 3.
            drop_p (float, optional): Dropout rate. Defaults to 0.3.
            activation (str, optional): Activation function. Defaults to "hardswish".
        """
        super().__init__()
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "drop_p": drop_p,
            "activation": activation,
            "initializer": initializer
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        layers = [            
            ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                stride=2,
                padding=(0, 0),
                bias=False,
                **factory_kwargs
            ),
            BatchNorm2d(in_channels, **factory_kwargs),
            get_activation(activation),
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=(1, 1),
                bias=False,
                **factory_kwargs
            ),
            BatchNorm2d(in_channels, **factory_kwargs),
            get_activation(activation),
            SeparableInvertResidual(in_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
        ]
        self.layers = Sequential(*layers)
        self._reset_parameters()
        

    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)
        self.initializer(self.layers[3].weight)
        if self.layers[0].bias is not None:
            ones_(self.layers[0].bias)
            ones_(self.layers[3].bias)


    def forward(
        self, 
        input: Tensor, 
    ) -> Tensor:
        Z = input
        return self.layers(Z)