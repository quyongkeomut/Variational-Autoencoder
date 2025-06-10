from typing import (
    Sequence,
    Callable
)
    
import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    GroupNorm,
    ConvTranspose2d
)

from neural_nets.activations import get_activation
from neural_nets.autoencoders.constituent_blocks import DownBlock, UpBlock

from utils.initializers import _get_initializer, ones_


PRE_H, PRE_W = 2, 2


class Encoder(Module):
    def __init__(
        self,
        img_channels: int,
        down_channels: Sequence[int],
        expand_factor: int,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        *args,
        **kwargs
    ):
        """
        Base Encoder

        Args:
            img_channels (int): Number of channels of input image. 
            down_channels (Sequence[int]): Sequence of channels along stages of Encoder
            expand_factor (int, optional): Coeficient which is used to expand number of channels.
            num_groups_norm (int, optional): Number of group for GN layer. Defaults to 4.
            activation (str, optional): Type of nonlinear activation function. 
                Defaults to "hardswish".
            initializer (str | Callable[[Tensor], Tensor], optional): Type of weight initializer. 
                Defaults to "he_uniform".
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.last_dim = down_channels[-1]
        self.pre_latent_dim = PRE_H * PRE_W * self.last_dim
        self.pre_latent_shape = (PRE_H, PRE_W)
        
        self.activation = activation
        self.factory_kwargs = factory_kwargs
        self.down_channels = down_channels
        num_stage = len(down_channels)
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation,
            "initializer": initializer
        }
        self.initializer = _get_initializer(initializer)
        
        # MAIN COMPONENTS
        
        # backbone
        layers = [
            # stem - stride = 0
            Sequential(
                Conv2d(
                    img_channels, 
                    down_channels[0]//2,
                    kernel_size=3,
                    padding=(1, 1),
                    **factory_kwargs
                ),
                get_activation(activation, **self.factory_kwargs),
            )
        ]
        # add stages
        down_channels_per_stage = list(down_channels) 
        down_channels_per_stage = [down_channels[0]//2] + down_channels_per_stage
        for stage_i in range(0, num_stage):
            layers.append(
                DownBlock(
                    in_channels=down_channels_per_stage[stage_i],
                    out_channels=down_channels_per_stage[stage_i + 1],
                    **invert_residual_kwargs,
                    **factory_kwargs
                )
            )
        self.layers = Sequential(*layers)
        # self._reset_parameters()
    
    def _reset_parameters(self):
        self.initializer(self.layers[0][0].weight)
        if self.layers[0][0].bias is not None:
            ones_(self.layers[0][0].bias)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    

class Decoder(Module):
    def __init__(
        self,
        img_channels: int,
        latent_channels: int,
        up_channels: Sequence[int],
        expand_factor: int,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        """
        Base Decoder

        Args:
            img_channels (int): Number of channels of input image
            latent_channels (int): _description_
            up_channels (Sequence[int]): Sequence of channels along stages of Decoder
            expand_factor (int, optional): Coeficient which is used to expand number of channels.
            num_groups_norm (int, optional): Number of group for GN layer. Defaults to 4.
            activation (str, optional): Type of nonlinear activation function. 
                Defaults to "hardswish".
            initializer (str | Callable[[Tensor], Tensor], optional): Type of weight initializer. 
                Defaults to "he_uniform".
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        num_stage = len(up_channels)
        
        self.latent_channels = latent_channels
        self.pre_latent_dim = PRE_H * PRE_W * self.latent_channels
        self.latent_shape = kwargs["latent_shape"]
        self.pre_latent_shape = (PRE_H, PRE_W)
        self.factory_kwargs = factory_kwargs
        self.img_channels = img_channels
        self.latent_channels = latent_channels
        self.initializer = _get_initializer(initializer)
        
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation,
            "initializer": initializer,
        }
        
        # MAIN COMPONENTS
        
        layers = []
        # latent conv to restore the topological
        # stem block - stride = 0
        projection = [
            ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=2,
                stride=2,
                padding=(0, 0),
                **self.factory_kwargs
            ),
            get_activation(activation, **self.factory_kwargs),
            Conv2d(latent_channels, up_channels[0], 1, **factory_kwargs),
            GroupNorm(num_groups_norm, up_channels[0], **factory_kwargs),
            get_activation(activation, **self.factory_kwargs)
        ]
        layers.extend(projection)
        
        # main layers
        up_channels_per_stage = list(up_channels) 
        up_channels_per_stage = up_channels_per_stage + [up_channels_per_stage[-1]//2]
        for stage_i in range(0, num_stage):
            layers.append(
                UpBlock(
                    up_channels[stage_i], 
                    up_channels_per_stage[stage_i + 1], 
                    **invert_residual_kwargs, 
                    **factory_kwargs
                )
            )
        # out layer
        layers.append(
            Sequential(
                Conv2d(
                    in_channels=up_channels[-1], 
                    out_channels=up_channels[-1], 
                    kernel_size=3, 
                    padding=(1, 1),
                    **factory_kwargs
                ),
                get_activation(activation, **self.factory_kwargs),
                # pointwise
                Conv2d(
                    up_channels[-1], 
                    img_channels, 
                    kernel_size=1, 
                    **factory_kwargs
                ),
            )
        )
        self.layers = Sequential(*layers)
        self._reset_parameters()
    
    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)   
        self.initializer(self.layers[2].weight)   
        self.initializer(self.layers[-1][0].weight)
        self.initializer(self.layers[-1][2].weight)        
        if self.layers[0].bias is not None:
            ones_(self.layers[0].bias)
            ones_(self.layers[2].bias)
            ones_(self.layers[-1][0].bias)
            ones_(self.layers[-1][2].bias)     
    
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.layers(input)