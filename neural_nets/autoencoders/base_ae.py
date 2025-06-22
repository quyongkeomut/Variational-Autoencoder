from typing import (
    Sequence,
    Callable,
    Optional
)
    
import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    GroupNorm,
    BatchNorm2d,
    ConvTranspose2d
)

from neural_nets.activations import get_activation
from neural_nets.autoencoders.constituent_blocks import DownBlock, UpBlock

from utils.initializers import _get_initializer, ones_, zeros_


LATENT_H, LATENT_W = 1, 1


class Encoder(Module):
    def __init__(
        self,
        img_channels: int,
        down_channels: Sequence[int],
        latent_dim: int,
        expand_factor: int,
        drop_p: float = 0.3,
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
            down_channels (Sequence[int]): Sequence of channels along stages of Encoder.
            latent_dim (int): Dimension of latent representation.
            expand_factor (int, optional): Coeficient which is used to expand number of channels.
            activation (str, optional): Type of nonlinear activation function. 
                Defaults to "hardswish".
            initializer (str | Callable[[Tensor], Tensor], optional): Type of weight initializer. 
                Defaults to "he_uniform".
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.last_dim = down_channels[-1]
        self.latent_dim = latent_dim
        self.down_channels = down_channels
        self.initializer = _get_initializer(initializer)
        self.activation = activation
        self.factory_kwargs = factory_kwargs
        self.invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "drop_p": drop_p,
            "activation": activation,
            "initializer": initializer,
        }
        # MAIN COMPONENTS
        self._set_backbone(img_channels, activation)
        Encoder._reset_parameters(self)


    def _set_backbone(
        self,
        img_channels,
        activation
    ):
        # backbone
        layers = [
            # stem - stride = 0
            Sequential(
                Conv2d(
                    img_channels, 
                    self.down_channels[0]//2,
                    kernel_size=3,
                    padding=(1, 1),
                    **self.factory_kwargs
                ),
                get_activation(activation, **self.factory_kwargs),
            )
        ]
        # add stages
        for idx, stage_i in enumerate(self.down_channels):
            layers.append(
                DownBlock(
                    in_channels=stage_i//2 if idx == 0 else self.down_channels[idx-1],
                    out_channels=stage_i,
                    **self.invert_residual_kwargs,
                    **self.factory_kwargs
                )
            )
        self.layers = Sequential(*layers)
    
    
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
        latent_dim: int,
        up_channels: Sequence[int],
        expand_factor: int,
        drop_p: float = 0.3,
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
            latent_dim (int): Dimension of latent representation.
            up_channels (Sequence[int]): Sequence of channels along stages of Decoder
            expand_factor (int, optional): Coeficient which is used to expand number of channels.
            drop_p (float, optional): Dropout rate. Defaults to 0.3.
            activation (str, optional): Type of nonlinear activation function. 
                Defaults to "hardswish".
            initializer (str | Callable[[Tensor], Tensor], optional): Type of weight initializer. 
                Defaults to "he_uniform".
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.latent_dim = latent_dim
        self.up_channels = up_channels
        self.latent_shape = (LATENT_H, LATENT_W)
        self.factory_kwargs = factory_kwargs
        self.img_channels = img_channels
        self.initializer = _get_initializer(initializer)
        self.activation = activation
        self.invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "drop_p": drop_p,
            "activation": activation,
            "initializer": initializer,
        }
        
        # MAIN COMPONENTS
        self._set_backbone(activation, img_channels)
        Decoder._reset_parameters(self)
        
        
    def _set_backbone(
        self,
        activation,
        img_channels,
    ):            
        layers = []
        projection = [
            ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=self.latent_dim,
                kernel_size=2,
                stride=2,
                padding=(0, 0),
                **self.factory_kwargs
            ),
            get_activation(self.activation, **self.factory_kwargs),
            Conv2d(
                in_channels=self.latent_dim,
                out_channels=self.latent_dim,
                kernel_size=1,
                padding=(0, 0),
                bias=False,
                **self.factory_kwargs
            ),
            BatchNorm2d(self.latent_dim, **self.factory_kwargs),
            get_activation(self.activation, **self.factory_kwargs),
        ]
        layers.extend(projection)
        
        # main layers
        for idx, stage_i in enumerate(self.up_channels):
            layers.append(
                UpBlock(
                    in_channels=self.latent_dim if idx == 0 else self.up_channels[idx-1], 
                    out_channels=stage_i, 
                    **self.invert_residual_kwargs, 
                    **self.factory_kwargs
                )
            )
        # out layer
        layers.append(
            Sequential(
                Conv2d(
                    in_channels=self.up_channels[-1], 
                    out_channels=self.up_channels[-1], 
                    kernel_size=3, 
                    padding=(1, 1),
                    **self.factory_kwargs
                ),
                get_activation(activation, **self.factory_kwargs),
                # projection
                Conv2d(
                    self.up_channels[-1], 
                    img_channels, 
                    kernel_size=3, 
                    padding=(1, 1),
                    **self.factory_kwargs
                ),
            )
        )
        self.layers = Sequential(*layers)
    
    
    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)   
        self.initializer(self.layers[2].weight)   
        self.initializer(self.layers[-1][0].weight)
        self.initializer(self.layers[-1][2].weight)        
        if self.layers[0].bias is not None:
            ones_(self.layers[0].bias)
            ones_(self.layers[-1][0].bias)
            zeros_(self.layers[-1][2].bias) 
        if self.layers[2].bias is not None:
            ones_(self.layers[2].bias)    
    
    
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.layers(input)
    
    
    @torch.no_grad()
    def sample(
        self,
    ):
        raise NotImplementedError
        
    
    def generate(self):
        raise NotImplementedError