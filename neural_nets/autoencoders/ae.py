from typing import (
    Sequence,
    Tuple,
    Callable
)
    
import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    GroupNorm,
    Linear
)
from torch.nn.functional import avg_pool2d

from neural_nets.conv_block import SeparableInvertResidual
from neural_nets.activations import get_activation
from neural_nets.autoencoders.constituent_blocks import DownBlock, UpBlock

from utils.initializers import _get_initializer, ones_


class Encoder(Module):
    def __init__(
        self,
        img_channels: int,
        down_channels: Sequence[int],
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        *args,
        **kwargs
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
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
        
        #
        # main components
        #
        
        # backbone
        layers = [
            # stem - stride = 0
            Sequential(
                Conv2d(
                    img_channels, 
                    down_channels[0],
                    kernel_size=1,
                    **factory_kwargs
                ),
                get_activation(activation),
            )
        ]
        
        # add stages
        down_channels_per_stage = list(down_channels) 
        down_channels_per_stage = [down_channels[0]] + down_channels_per_stage
        for stage_i in range(0, num_stage):
            layers.append(
                DownBlock(
                    in_channels=down_channels_per_stage[stage_i],
                    out_channels=down_channels_per_stage[stage_i + 1],
                    **invert_residual_kwargs,
                    **factory_kwargs
                )
            )
        
        # latent layer
        layers.append(
            SeparableInvertResidual(
                down_channels_per_stage[stage_i + 1], 
                down_channels_per_stage[stage_i + 1], 
                **invert_residual_kwargs, **factory_kwargs),
        )
        
        self.layers = Sequential(*layers)
        # self._reset_parameters()
    
    
    def _reset_parameters(self):
        self.initializer(self.layers[0][0].weight)
        
        if self.layers[0][0].bias is not None:
            ones_(self.layers[0][0].bias)


    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    

class VAEEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_dim = self.down_channels[-1] 
        H, W = kwargs["latent_shape"]
        self.lattent_dim = H*W * self.last_dim
        
        # append the µ and σ2 layers
        self.mean = Sequential(
            Linear(
                in_features=self.last_dim,
                out_features=self.lattent_dim,
                **self.factory_kwargs
            ),
        )
        self.log_var = Sequential(
            Linear(
                in_features=self.last_dim,
                out_features=self.lattent_dim,
                **self.factory_kwargs
            ),
        )
        self._reset_parameters()
    
    
    def _reset_parameters(self):
        super()._reset_parameters()
        self.initializer(self.mean[0].weight)
        self.initializer(self.log_var[0].weight)
    
        ones_(self.mean[0].bias)
        ones_(self.log_var[0].bias)
    

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        Z = self.layers(input)
        N, C = Z.shape[:2]
        Z = avg_pool2d(Z, Z.shape[2:]).view(N, C) # (N, C)
        mean = self.mean(Z)
        log_var = self.log_var(Z)
        return mean, log_var
    
   
class Decoder(Module):
    def __init__(
        self,
        img_channels: int,
        latent_channels: int,
        up_channels: Sequence[int],
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        num_stage = len(up_channels)
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation,
            "initializer": initializer,
        }
        self.img_channels = img_channels
        self.latent_channels = latent_channels
        self.initializer = _get_initializer(initializer)
        
        #
        # main components
        #
        
        # add stages
        layers = []
        
        # stem block - stride = 0
        projection = [
            Conv2d(latent_channels, up_channels[0], 1, **factory_kwargs),
            GroupNorm(num_groups_norm, up_channels[0], **factory_kwargs)
        ]
        layers.extend(projection)
        
        # main layers
        up_channels_per_stage = list(up_channels) 
        up_channels_per_stage = up_channels_per_stage + [up_channels_per_stage[-1]]
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
                Conv2d(up_channels[-1], img_channels, kernel_size=1, **factory_kwargs),
                # get_activation("sigmoid")
            )
        )
        
        self.layers = Sequential(*layers)
        self._reset_parameters()
        
    
    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)   
        self.initializer(self.layers[-1][0].weight)        
        if self.layers[0].bias is not None:
            ones_(self.layers[0].bias)
            ones_(self.layers[-1][0].bias)     
    
    
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.layers(input)
    
    
class VAEDecoder(Decoder):
    def __init__(
        self, 
        std_dec: float = 1.,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.std_dec = std_dec
        self.latent_shape = kwargs["latent_shape"]
    
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward method of VAE Decoder

        Args:
            input (Tensor): Latent vector

        Shape:
            input: (N, d)
        
        Returns:
            Tensor: Reconstructed image
        """
        Z = input.view(input.size(0), self.latent_channels, *self.latent_shape) # (N, C, H, W)
        return self.layers(Z)
