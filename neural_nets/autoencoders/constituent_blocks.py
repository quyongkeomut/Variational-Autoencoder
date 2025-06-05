from typing import Callable

import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    Upsample,
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
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
    ):  
        r"""
        Downsample block of lightweight UNet

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            expand_factor (int, optional): Expand factor used in expansion conv layer
                of inverted residual block . Defaults to 3.
            num_groups_norm (int, optional): Number of group to be normalized by group norm. 
                Defaults to 4.
            activation (str, optional): Activation function. Defaults to "hardswish".
        """
        super().__init__()
        
        invert_residual_kwargs = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation,
            "initializer": initializer
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        layers = [       
            # depthwise conv for downsampling 
            Conv2d(
                in_channels, 
                in_channels,
                kernel_size=3,
                stride=2,
                padding=(1, 1),
                groups=in_channels,
                **factory_kwargs
            ), 
            get_activation(activation),
            
            # pointwise conv for projection
            Conv2d(
                in_channels, 
                out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            get_activation(activation),
            
            # main layers
            SeparableInvertResidual(**invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(**invert_residual_kwargs, **factory_kwargs),
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
        r"""
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
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
    ):  
        super().__init__()
        
        invert_residual_kwargs = {
            "in_channels": out_channels,
            "out_channels": out_channels,
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation,
            "initializer": initializer
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(initializer)
        
        layers = [
            # pointwise conv for projection
            Conv2d(
                in_channels, 
                out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            get_activation(activation),
            
            # depthwise conv
            Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1),
                groups=out_channels,
                **factory_kwargs
            ),
            get_activation(activation),
            
            Upsample(scale_factor=2, mode="bilinear"),
            SeparableInvertResidual(**invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(**invert_residual_kwargs, **factory_kwargs),
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
        input: Tensor, 
    ) -> Tensor:
        Z = input
        return self.layers(Z)